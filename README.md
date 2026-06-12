# 台股投資組合當沖回測系統

一套以 Python 撰寫的台股當沖（Day Trading）回測框架，支援**多檔股票投資組合**回測。
資料透過 [FinMind](https://finmind.github.io/) 抓取，回測引擎完整計入台股**手續費、當沖交易稅、滑價與停損**，並產出組合層級與個股層級的績效指標與資金曲線。

**設計核心：策略與回測引擎完全分離。** 你只需要產生一個格式正確的「訊號矩陣」，
就能直接餵進引擎回測——策略邏輯寫在 `signals.py`，引擎完全不需更動。

---

## 檔案結構

```
project/
├── data_loader.py        # 資料載入：兩種抓取（快取 load_data/load_multiple、直抓 get_ohlcv）
├── portfolio_backtest.py # 回測引擎：PortfolioBacktester
├── signals.py            # ★ 你寫策略的地方（產生訊號矩陣）
├── main.py               # 範例主程式：載入 → 訊號 → 回測 → 報表/繪圖
├── backtest.py            # 單股當沖回測引擎
├── strategy.py            # 單股 5/20 日均線交叉策略
├── requirements.txt       # Python 相依套件
├── data/                 # 自動建立，存放各股票 CSV 快取
├── .env                  # FINMIND_TOKEN=...（自行填入，勿提交 git）
└── .env.example          # .env 範本
```

---

## 安裝與設定

需要 Python 3.9+。安裝相依套件：

```bash
python -m pip install -r requirements.txt
```

設定 FinMind token（到 [finmindtrade.com](https://finmindtrade.com/) 註冊取得）：

```bash
cp .env.example .env
# 編輯 .env，填入：FINMIND_TOKEN=你的_token
```

執行範例回測：

```bash
python main.py
```

> 首次執行會從 FinMind 抓資料並存成 `data/{股票代號}.csv`；之後再跑會直接讀快取，不再連網。
> 想重新抓取，刪掉對應的 CSV 即可。

---

## 資料抓取機制（兩種方式）

本系統提供**兩種抓取資料的方式**，由 `data_loader.py` 實作：

**怎麼選？** 一次拉一群股票用 `load_multiple`、單一檔用 `load_data`（兩者都帶快取）；
要繞過快取強制向 FinMind 直抓最新資料用 `get_ohlcv`。

### 方式一：本地 CSV 快取（`load_data` / `load_multiple`）★ 預設

帶快取的智慧載入，是 `main.py` 使用的方式：

1. 若 `data/{stock_id}.csv` **存在且已完整涵蓋**指定期間 → **直接讀檔，不連網**，回傳裁切至指定範圍的資料。
2. 若快取**存在但範圍不足**（缺前段或後段）→ 向 FinMind **補抓**指定期間，與既有快取**合併、去重**後重新存檔。
3. 若快取**不存在** → 從 FinMind 抓取，自動建立 `data/` 資料夾並存成 CSV。

```python
from data_loader import load_data, load_multiple

# 單檔（帶快取）
df = load_data("2330", "2023-01-01", "2024-12-31")

# 多檔批次（帶快取），回傳 dict[str, pd.DataFrame]
data = load_multiple(["2330", "2317", "0050"], "2023-01-01", "2024-12-31")
```

優點：重複回測時不重抓、不耗 FinMind 流量（免費額度 600 次/小時）。

### 方式二：FinMind API 直抓（`get_ohlcv`）

**不經快取**，每次都直接向 FinMind 拉取最新資料。適合需要繞過 CSV、強制取得最新資料的場合：

```python
from data_loader import get_ohlcv

# 純 FinMind 直抓，不讀也不寫本地 CSV
df = get_ohlcv("2330", "2023-01-01", "2024-12-31")
```

> 兩種方式回傳的都是**標準 OHLCV DataFrame**（index 為 `DatetimeIndex`，
> 欄位 `open/high/low/close/volume` 皆為 `float`）。
> `load_data` 內部在快取不足時其實也是呼叫 `get_ohlcv` 來補抓。

---

## 如何撰寫自己的策略

**所有策略邏輯都寫在 `signals.py`。** 你不需要碰回測引擎。

`signals.py` 預設的 `generate_signals_matrix()` 回傳一個**全 0 矩陣（不交易）**，
只是把流程接通。請打開檔案，在 `TODO` 處填入你的邏輯。

### 訊號矩陣（signals）格式規範

`generate_signals_matrix(data)` 必須回傳一個 `pd.DataFrame`：

| 項目 | 規範 |
|------|------|
| 型態 | `pd.DataFrame` |
| `index` | `datetime`（所有股票交易日的**聯集**） |
| `columns` | 股票代號字串，例如 `"2330"` |
| 值 | `1`（買入/做多）、`-1`（賣出/做空）、`0`（不動作） |
| 缺值 | 一律填 `0`；某股票某日無資料時該格視為不交易 |

範例矩陣長這樣：

```
            2330  2317  2454
date
2023-01-03     0     1     0
2023-01-04     1     0    -1
2023-01-05     0     0     0
```

### 撰寫範本

```python
import pandas as pd

def generate_signals_matrix(data: dict) -> pd.DataFrame:
    all_dates = sorted(set().union(*(df.index for df in data.values())))
    matrix = pd.DataFrame(0, index=pd.DatetimeIndex(all_dates),
                          columns=list(data.keys()), dtype=int)

    for stock_id, df in data.items():
        my_signal = your_logic(df)        # 回傳 pd.Series，值為 1/-1/0
        matrix.loc[my_signal.index, stock_id] = my_signal.astype(int)

    return matrix.fillna(0).astype(int)
```

`signals.py` 底部附了一個**「5/20 日均線交叉」參考範例**（預設註解停用），
想找起點的話可以解除註解直接使用。

> **避免未來函數（look-ahead bias）：** 引擎以「當日開盤價」進場。
> 若你用「當日收盤」算訊號，等於用了當天才知道的資訊去當天開盤下單。
> 範例策略已用 `shift(1)` 把訊號延後一天（T-1 收盤算、T 開盤執行）來避免這問題。

---

## 回測引擎規則

### 交易成本（台股規則，全部可在初始化時調整）

| 項目 | 預設值 | 說明 |
|------|--------|------|
| 手續費 | `0.001425 × 0.6` | 六折，買賣**雙邊**都收 |
| 交易稅 | `0.003 × 0.2 = 0.0006` | 當沖二折，**僅賣出**收 |

做多時賣出在出場收稅；做空時賣出在進場收稅。

### 滑價與停損（逐股票獨立計算）

| 項目 | 預設值 |
|------|--------|
| 停損百分比 | `2%` |
| 滑價百分比 | `0.1%` |
| 每筆股數 | `1000`（1 張） |

- **進場**：以當日**開盤價**計算（做多加滑價、做空減滑價）。
- **停損**：做多用當日 `low` 判斷、做空用當日 `high` 判斷觸發；觸發則以停損價出場。
- **出場**：未觸停損則以當日**收盤價**（含滑價）出場。

---

## API 速查

### `data_loader.py`

```python
load_data(stock_id, start_date, end_date) -> pd.DataFrame
    # 方式一：載入單檔，優先讀本地 CSV，範圍不足才向 FinMind 補抓並快取

load_multiple(stock_ids, start_date, end_date) -> dict[str, pd.DataFrame]
    # 方式一（批次）：多檔載入，回傳以股票代號為 key 的字典

get_ohlcv(stock_id, start_date, end_date) -> pd.DataFrame
    # 方式二：不經快取，直接從 FinMind API 抓取
```

### `portfolio_backtest.py` — `PortfolioBacktester`

```python
bt = PortfolioBacktester(
    fee_rate=0.001425, fee_discount=0.6,
    tax_rate=0.003, day_trade_tax_ratio=0.2,
    stop_loss_pct=0.02, slippage_pct=0.001, shares=1000,
)

bt.run(data, signals)      # -> 交易明細 DataFrame
bt.daily_summary()         # -> 每日 daily_pnl / cumulative_pnl / n_positions / n_stop_loss
bt.report()                # -> 印出並回傳績效指標 dict（含個股績效表）
bt.plot_equity_curve()     # -> 畫資金曲線 + 每日曝險圖
```

**`report()` 指標：** 總損益、總交易筆數、整體勝率、年化 Sharpe Ratio（基於每日報酬率，252 交易日）、最大回撤、平均每日持倉數、總停損觸發次數、個股層級績效表（每股筆數/總損益/勝率）。

---

## 注意事項

- **等張數而非等金額**：每筆固定 1000 股，不同價位股票的實際曝險差異很大（例如 2330 一張約 60 萬、0050 一張約 4 萬）。若需等金額加權，請自行調整。
- **Sharpe 報酬率分母**：以「當日名目曝險（進場價 × 股數之總和）」為分母計算每日報酬率。
- `.env` 含個人 token，請務必加入 `.gitignore`，不要提交到 git。
# Strategy-System
