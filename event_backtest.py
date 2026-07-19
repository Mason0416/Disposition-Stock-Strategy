"""處置股做多——進場時機事件回測。

讀取 data/disposition_events_clean.csv 與個股日K，對每一筆處置事件測試
「處置期間第幾天進場」對績效的影響。進場後逐日以動態停損線檢查盤中最低價，
未觸發則抱到 period_end 收盤出場。

輸出 data/trade_level.csv（事件 x 進場日 的完整明細）與各種績效彙總表。

用法：
    python event_backtest.py
"""

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import fetch_trading_calendar, load_data

# 交易成本參數
FEE_RATE = 0.001425          # 券商手續費率（單邊）
FEE_DISCOUNT = 0.2           # 手續費折扣（二折）
TAX_RATE = 0.003             # 證交稅率（跨日持倉，全額 0.3%，無當沖優惠）
SLIPPAGE_PCT = 0.001         # 基準滑價：進場價往上加、出場價往下減
STOP_LOSS_PCT = 0.09         # 停損百分比（以前一日收盤價為基準動態計算）

# 單邊有效手續費率
EFFECTIVE_FEE = FEE_RATE * FEE_DISCOUNT

# 滑價敏感度測試情境
SLIPPAGE_SCENARIOS = (0.001, 0.005, 0.01)

# 停損成交價模式：
#   "stop_line"    —— 以當日停損線成交（規格指定；跳空跌破時偏樂觀）
#   "gap_adjusted" —— 以 min(停損線, 當日開盤價) 成交，反映跳空跌破無法
#                     在停損線成交的情況
STOP_FILL_MODES = ("stop_line", "gap_adjusted")

# 每筆交易的固定本金（元）；連續金額試算，不處理張數／零股取整
CAPITAL_NTD = 1_000_000

# 年化交易日數
TRADING_DAYS_PER_YEAR = 252

# 股價抓取時，在事件區間前後各補幾個交易日
PRICE_PAD_DAYS = 5

CLEAN_CSV = "data/disposition_events_clean.csv"
TRADE_CSV = "data/trade_level.csv"
BASELINE_CSV = "data/baseline_summary.csv"

# =========================================================================
# BASELINE —— 目前選定的正式版本
#
# 這是選定的單一設定，不是掃描/敏感度測試的其中一組。以下參數已鎖定，
# 要看悲觀情境請另跑 SLIPPAGE_SCENARIOS，勿修改此處。
#
# 注意一：進場日的差異在統計上不顯著。開盤價進場、無停損下，第 2~6 天的
# 平均報酬彼此差距都落在雜訊範圍內（單筆報酬標準差約 13%，遠大於組間差異）。
# 選第 2 天不是因為它被證明最優。可確定的只有：此中段區帶優於第 1 天與
# 第 9 天之後（曝險不足）。
#
# 注意二：改用第 2 天的理由——整體樣本下第 2 天平均報酬與總 pnl 略高於第 5 天
# （4.00% vs 3.65%、總 pnl +9.6%），且三組分級掃描中，各組最佳進場日都落在
# 第 2~3 天（第一次第 2 天、接續升級第 2 天、獨立20分第 3 天），第 2 天同時
# 貼近各組最佳點。代價是持有天數較長（約 9 天 vs 6 天）、資金佔用較久。
#
# 注意三：最佳進場日會隨「進場價格基準」改變。收盤價進場版本最佳點偏後
# （約第 3~4 天），開盤價進場版本偏前。原因是處置期間的動能：早進場時
# 「當天開盤價比前一天收盤貴」的懲罰在早期最重，隨進場日遞減。本 baseline
# 採開盤價進場。
#
# 注意四：本 baseline 為「無停損」（stop_loss_pct=None，持有至 period_end
# 收盤）。原 9% 動態停損版本經配對檢定顯示會提早砍掉回檔的贏家，整體報酬
# 反而較低（於第 5 天進場下比較：平均 2.10% vs 無停損 3.65%），代價僅是
# 尾部風險較大。9% 停損版本的完整數字與分析移至 RESEARCH.md 保留。
# stop_from_entry_day / stop_fill_mode 在無停損下不生效，保留僅為相容。
# =========================================================================
BASELINE_CONFIG = {
    "entry_day_index": 2,
    "entry_price_mode": "open",
    "stop_loss_pct": None,             # None = 無停損，持有至 period_end
    "stop_from_entry_day": True,       # 無停損下不生效（保留相容）
    "stop_fill_mode": "stop_line",     # 無停損下不生效（保留相容）
    "slippage_pct": 0.001,
    "fee_rate": FEE_RATE,
    "fee_discount": FEE_DISCOUNT,
    "tax_rate": TAX_RATE,
    "capital_ntd": CAPITAL_NTD,
}

# 樣本數的性質說明，隨摘要一起輸出
SAMPLE_NOTE = (
    "為「已完成事件」數量（clean pipeline 的 L6 已排除 period_end 尚未到期、"
    "處置仍在進行中者）。此數字之後只會因為新的處置事件發生而變多，"
    "不會因為既有事件突然補到出場價而變動——後者曾使 baseline 在未改動"
    "程式的情況下自行從 2,344 筆漂移到 2,349 筆。"
)

# Sharpe 暫不提供的原因，隨摘要一起輸出，避免日後被誤填
SHARPE_CAVEAT = (
    "暫不提供。現行算法將平均約 6 個交易日的持有報酬當成單日報酬乘上 "
    "sqrt(252) 年化，且每個出場日跨事件平均稀釋了標準差、無出場的日子"
    "不計入序列，三者都會灌大數值（曾算出 5.47 的不合理結果）。"
    "需建立涵蓋每個交易日、計入未平倉部位市值變動的完整每日權益曲線"
    "才能得出可信數字，此事尚未完成。"
)


def _section(title: str) -> None:
    """印出區段標題分隔線。"""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def compute_return(entry_price: float, exit_price: float,
                   slippage_pct: float = SLIPPAGE_PCT) -> float:
    """計算單筆做多交易扣除成本後的淨報酬率。

    進場價含滑價往上、出場價含滑價往下；進出場皆收手續費，
    證交稅只在出場收一次。

    Args:
        entry_price: 進場成交參考價。
        exit_price: 出場成交參考價。
        slippage_pct: 滑價百分比。

    Returns:
        淨報酬率（小數，例如 0.05 表示 +5%）。
    """
    entry_exec = entry_price * (1 + slippage_pct)
    exit_exec = exit_price * (1 - slippage_pct)

    cost_in = entry_exec * (1 + EFFECTIVE_FEE)
    proceeds = exit_exec * (1 - EFFECTIVE_FEE - TAX_RATE)
    return proceeds / cost_in - 1


def build_price_ranges(events: pd.DataFrame,
                       calendar: pd.DatetimeIndex) -> dict:
    """為每檔股票計算需要抓取的日K日期範圍（多次事件取聯集）。

    範圍為該檔最早 period_start 往前 PRICE_PAD_DAYS 個交易日，
    到最晚 period_end 往後 PRICE_PAD_DAYS 個交易日。

    Args:
        events: 處置事件 DataFrame。
        calendar: 交易日曆。

    Returns:
        dict[str, tuple[str, str]]，key 為 stock_id，
        value 為 (start_date, end_date) 字串。
    """
    ranges = {}
    for stock_id, group in events.groupby("stock_id"):
        first = group["period_start"].min()
        last = group["period_end"].max()

        i = calendar.searchsorted(first, side="left")
        j = calendar.searchsorted(last, side="right") - 1
        start = calendar[max(0, i - PRICE_PAD_DAYS)]
        end = calendar[min(len(calendar) - 1, j + PRICE_PAD_DAYS)]

        ranges[stock_id] = (start.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"))
    return ranges


def load_prices(ranges: dict) -> dict:
    """逐檔載入日K（沿用 load_data 的本地 CSV 快取）。

    保留完整 OHLC：停損觸發需要盤中最低價，跳空調整需要開盤價。
    同一檔只抓一次；單檔失敗時印出警告並略過。

    Args:
        ranges: build_price_ranges 產生的日期範圍字典。

    Returns:
        dict[str, pd.DataFrame]，key 為 stock_id，value 為標準 OHLCV
        DataFrame。
    """
    prices = {}
    failed = []
    total = len(ranges)

    for i, (stock_id, (start, end)) in enumerate(sorted(ranges.items()), 1):
        if i % 200 == 0 or i == total:
            print(f"  [{i}/{total}] 載入中…")
        try:
            prices[stock_id] = load_data(stock_id, start, end)
        except Exception as exc:  # noqa: BLE001
            failed.append((stock_id, str(exc)[:60]))

    if failed:
        print(f"[警告] {len(failed)} 檔載入失敗，已略過：")
        for stock_id, msg in failed[:10]:
            print(f"  - {stock_id}: {msg}")
        if len(failed) > 10:
            print(f"  …另有 {len(failed) - 10} 檔")
    return prices


def simulate_trade(ohlc: pd.DataFrame, period_days: pd.DatetimeIndex,
                   entry_pos: int, stop_loss_pct: float,
                   stop_fill_mode: str,
                   entry_price_mode: str = "close",
                   stop_from_entry_day: bool = False,
                   stop_basis: str = "trailing_daily") -> dict:
    """模擬單筆做多交易，進場後逐日檢查盤中最低價是否觸及停損線。

    停損線的計算方式由 stop_basis 決定：
      "trailing_daily"（預設，維持既有 baseline 行為）：每日重算，
          stop_line(t) = close(t-1) * (1 - stop_loss_pct)，進場隔日的
          close(t-1) 即進場日收盤價。
      "entry_fixed"：全程固定，stop_line = entry_price * (1 - stop_loss_pct)。

    兩種基準下，觸發判斷（當日 low <= stop_line）與成交邏輯完全相同；
    一路未觸發則抱到 period_end 收盤。停牌（無資料）的日子跳過不檢查，
    亦不更新前一日收盤價。

    Args:
        ohlc: 該檔標準 OHLCV DataFrame。
        period_days: 該事件處置期間的交易日。
        entry_pos: 進場日在 period_days 中的位置（0-based）。
        stop_loss_pct: 停損百分比。
        stop_fill_mode: 停損成交價模式，見 STOP_FILL_MODES。
        entry_price_mode: 進場價取用 "close"（收盤）或 "open"（開盤）。
        stop_from_entry_day: 是否自進場日當天起就檢查停損。開盤價進場時
            當日盤中風險為真，設為 True 才反映實際；收盤價進場時當日
            已無盤中風險，應維持 False。
        stop_basis: 停損線基準，"trailing_daily"（每日重算）或
            "entry_fixed"（進場價固定），見上方說明。

    Returns:
        含 entry_price, exit_price, exit_date, exit_reason, gapped 的 dict；
        進場日無價則回傳 None。
    """
    entry_date = period_days[entry_pos]
    if entry_date not in ohlc.index:
        return None
    entry_price = float(ohlc.loc[entry_date, entry_price_mode])
    if entry_price <= 0:
        return None

    # stop_loss_pct 為 None：完全不套用停損，直接持有到 period_end。
    if stop_loss_pct is None:
        exit_date = period_days[-1]
        return {
            "entry_price": entry_price,
            "exit_price": float(ohlc.loc[exit_date, "close"]),
            "exit_date": exit_date,
            "exit_reason": "period_end",
            "gapped": False,
        }

    # trailing_daily 以「前一交易日收盤價」為基準（與進場價取法無關）；
    # entry_fixed 全程固定為進場價乘上停損比例。
    check_days = period_days[entry_pos:] if stop_from_entry_day \
        else period_days[entry_pos + 1:]

    if stop_from_entry_day:
        # 進場日的停損線需要更前一天的收盤價；取不到則退回隔日起檢查
        if entry_pos == 0 or period_days[entry_pos - 1] not in ohlc.index:
            check_days = period_days[entry_pos + 1:]
            prev_close = float(ohlc.loc[entry_date, "close"])
        else:
            prev_close = float(ohlc.loc[period_days[entry_pos - 1], "close"])
    else:
        prev_close = float(ohlc.loc[entry_date, "close"])

    fixed_stop_line = entry_price * (1 - stop_loss_pct)

    for day in check_days:
        if day not in ohlc.index:
            continue
        row = ohlc.loc[day]
        if stop_basis == "entry_fixed":
            stop_line = fixed_stop_line
        else:
            stop_line = prev_close * (1 - stop_loss_pct)

        if float(row["low"]) <= stop_line:
            open_price = float(row["open"])
            gapped = open_price < stop_line
            # 進場當日即觸發時，開盤已在停損線下方代表我們是在停損線之下
            # buy 進場的，不可能再以停損線賣出（否則停損反而記成獲利），
            # 故一律以開盤價成交。
            force_open = gapped and day == entry_date
            if gapped and (stop_fill_mode == "gap_adjusted" or force_open):
                exit_price = open_price
            else:
                exit_price = stop_line
            return {
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "exit_date": day,
                "exit_reason": "stop_loss",
                "gapped": gapped,
            }
        prev_close = float(row["close"])

    # 未觸發停損：抱到 period_end 收盤
    exit_date = period_days[-1]
    return {
        "entry_price": entry_price,
        "exit_price": float(ohlc.loc[exit_date, "close"]),
        "exit_date": exit_date,
        "exit_reason": "period_end",
        "gapped": False,
    }


def build_trade_level(events: pd.DataFrame, prices: dict,
                      calendar: pd.DatetimeIndex,
                      slippage_pct: float = SLIPPAGE_PCT,
                      stop_loss_pct: float = STOP_LOSS_PCT,
                      stop_fill_mode: str = "stop_line",
                      entry_price_mode: str = "close",
                      stop_from_entry_day: bool = False,
                      stop_basis: str = "trailing_daily",
                      verbose: bool = True) -> pd.DataFrame:
    """建立「事件 x 進場日」的交易明細。

    對每筆事件，將 period_start ~ period_end 的交易日依序標記
    entry_day_index = 1..trading_days，只取 1..trading_days-1 為候選進場日
    （最後一天當天進當天出無意義）。停牌／無資料的交易日直接跳過，
    不以前一日價格替代。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。
        slippage_pct: 滑價百分比。
        stop_loss_pct: 停損百分比。
        stop_fill_mode: 停損成交價模式。
        entry_price_mode: 進場價取用 "close" 或 "open"。
        stop_from_entry_day: 是否自進場日當天起就檢查停損。
        stop_basis: 停損線基準，"trailing_daily" 或 "entry_fixed"。
        verbose: 是否印出略過統計。

    Returns:
        交易明細 DataFrame。
    """
    records = []
    skipped_no_stock = 0
    skipped_no_exit = 0
    skipped_no_price = 0

    for _, event in events.iterrows():
        stock_id = event["stock_id"]
        ohlc = prices.get(stock_id)
        if ohlc is None:
            skipped_no_stock += 1
            continue

        start = event["period_start"]
        end = event["period_end"]
        i = calendar.searchsorted(start, side="left")
        j = calendar.searchsorted(end, side="right")
        period_days = calendar[i:j]

        if end not in ohlc.index:
            skipped_no_exit += 1
            continue

        n = int(event["trading_days"])
        for idx in range(1, n):
            trade = simulate_trade(ohlc, period_days, idx - 1,
                                   stop_loss_pct, stop_fill_mode,
                                   entry_price_mode, stop_from_entry_day,
                                   stop_basis)
            if trade is None:
                skipped_no_price += 1
                continue

            holding = calendar.searchsorted(trade["exit_date"], side="right") \
                - calendar.searchsorted(period_days[idx - 1], side="left")
            records.append({
                "stock_id": stock_id,
                "stock_name": event["stock_name"],
                "market": event["market"],
                "condition_category": event["condition_category"],
                "disposition_order": event["disposition_order"],
                "period_start": start,
                "period_end": end,
                "trading_days": n,
                "entry_day_index": idx,
                "days_until_exit": n - idx,
                "entry_date": period_days[idx - 1],
                "exit_date": trade["exit_date"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "holding_days": int(holding),
                "exit_reason": trade["exit_reason"],
                "gapped": trade["gapped"],
                "return_pct": compute_return(trade["entry_price"],
                                             trade["exit_price"],
                                             slippage_pct),
            })
            records[-1]["pnl_ntd"] = CAPITAL_NTD * records[-1]["return_pct"]

    if verbose:
        print(f"  略過：該檔無日K {skipped_no_stock} 筆事件、"
              f"出場日無價 {skipped_no_exit} 筆事件、"
              f"進場日停牌 {skipped_no_price} 筆(事件x進場日)")
    return pd.DataFrame(records)


def summarize(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """依指定欄位分組計算績效統計。

    Args:
        trades: 交易明細 DataFrame。
        by: 分組欄位名稱。

    Returns:
        績效彙總 DataFrame，報酬率相關欄位以百分比表示。
    """
    grouped = trades.groupby(by)["return_pct"]
    stop_rate = trades.groupby(by)["exit_reason"].apply(
        lambda s: (s == "stop_loss").mean() * 100
    )
    pnl = trades.groupby(by)["pnl_ntd"]
    # 每個分組是獨立的策略版本，各自有自己的每日損益序列
    sharpe = pd.Series(
        {key: compute_sharpe(group) for key, group in trades.groupby(by)},
        name="Sharpe",
    )
    out = pd.DataFrame({
        "樣本數": grouped.size(),
        "勝率%": (grouped.apply(lambda s: (s > 0).mean()) * 100).round(1),
        "停損%": stop_rate.round(1),
        "平均%": (grouped.mean() * 100).round(2),
        "中位數%": (grouped.median() * 100).round(2),
        "標準差%": (grouped.std() * 100).round(2),
        "平均pnl_ntd": pnl.mean().round(0),
        "總pnl_ntd": pnl.sum().round(0),
        "Sharpe": sharpe.round(3),
        "P10%": (grouped.quantile(0.10) * 100).round(2),
        "P90%": (grouped.quantile(0.90) * 100).round(2),
    })
    return out


def apply_slippage(trades: pd.DataFrame, slippage_pct: float) -> pd.DataFrame:
    """以指定滑價重算交易明細的報酬率。

    停損觸發與否只取決於 low 與停損線（皆與滑價無關），故同一份明細
    可直接換算不同滑價，不需重跑模擬。

    Args:
        trades: 交易明細 DataFrame。
        slippage_pct: 滑價百分比。

    Returns:
        report_pct 已依指定滑價重算的 DataFrame 複本。
    """
    out = trades.copy()
    entry_exec = out["entry_price"] * (1 + slippage_pct)
    exit_exec = out["exit_price"] * (1 - slippage_pct)
    cost_in = entry_exec * (1 + EFFECTIVE_FEE)
    proceeds = exit_exec * (1 - EFFECTIVE_FEE - TAX_RATE)
    out["return_pct"] = proceeds / cost_in - 1
    out["pnl_ntd"] = CAPITAL_NTD * out["return_pct"]
    return out


def compute_sharpe(trades: pd.DataFrame) -> float:
    """以每日實現損益／每日名目曝險計算年化 Sharpe Ratio。

    比照 portfolio_backtest.py 的算法：先建立每日報酬率序列
    （當日損益 / 當日名目曝險），再取 平均/標準差 x sqrt(252)。
    不使用「單筆報酬率平均/標準差」的簡化算法。

    交易依 exit_date 分組（同一天多筆出場則加總），當日名目曝險為
    當天出場筆數 x CAPITAL_NTD。無出場的日子不構成序列點。

    Args:
        trades: 含 exit_date 與 pnl_ntd 的交易明細。

    Returns:
        年化 Sharpe Ratio；序列不足或標準差為 0 時回傳 0.0。
    """
    if trades.empty:
        return 0.0

    daily = trades.groupby("exit_date")["pnl_ntd"].agg(["sum", "size"])
    daily_notional = daily["size"] * CAPITAL_NTD
    daily_ret = daily["sum"] / daily_notional

    std = daily_ret.std()
    if std and std > 0:
        return float(daily_ret.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))
    return 0.0


def scenario_stats(trades: pd.DataFrame) -> dict:
    """計算單一情境的整體績效指標。

    Args:
        trades: 交易明細 DataFrame。

    Returns:
        績效指標 dict。
    """
    ret = trades["return_pct"]
    stopped = (trades["exit_reason"] == "stop_loss")
    return {
        "樣本數": len(trades),
        "平均%": round(ret.mean() * 100, 2),
        "中位數%": round(ret.median() * 100, 2),
        "勝率%": round((ret > 0).mean() * 100, 1),
        "標準差%": round(ret.std() * 100, 2),
        "平均pnl_ntd": round(trades["pnl_ntd"].mean()),
        "總pnl_ntd": round(trades["pnl_ntd"].sum()),
        "Sharpe": round(compute_sharpe(trades), 3),
        "停損次數": int(stopped.sum()),
        "停損比例%": round(stopped.mean() * 100, 1),
        "平均持有天數": round(trades["holding_days"].mean(), 2),
    }


def run_baseline(events: pd.DataFrame, prices: dict,
                 calendar: pd.DatetimeIndex) -> tuple:
    """執行 BASELINE 正式版本，回傳交易明細與績效摘要。

    參數一律取自 BASELINE_CONFIG，不接受覆寫——此函式的用途是產出
    「目前選定版本」的固定結果，掃描與敏感度測試請用 build_trade_level。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。

    Returns:
        (trades, summary) — 交易明細 DataFrame 與績效摘要 dict。
    """
    cfg = BASELINE_CONFIG
    all_trades = build_trade_level(
        events, prices, calendar,
        slippage_pct=cfg["slippage_pct"],
        stop_loss_pct=cfg["stop_loss_pct"],
        stop_fill_mode=cfg["stop_fill_mode"],
        entry_price_mode=cfg["entry_price_mode"],
        stop_from_entry_day=cfg["stop_from_entry_day"],
        verbose=False,
    )
    trades = all_trades[
        all_trades["entry_day_index"] == cfg["entry_day_index"]
    ].copy()

    ret = trades["return_pct"]
    summary = {
        "樣本數": len(trades),
        "樣本說明": SAMPLE_NOTE,
        "停損機制": "無（持有至 period_end 收盤）",
        "勝率%": round((ret > 0).mean() * 100, 1),
        "平均報酬率%": round(ret.mean() * 100, 2),
        "中位數報酬率%": round(ret.median() * 100, 2),
        "報酬率標準差%": round(ret.std() * 100, 2),
        "平均pnl_ntd": round(trades["pnl_ntd"].mean()),
        "總pnl_ntd": round(trades["pnl_ntd"].sum()),
        "平均持有天數": round(trades["holding_days"].mean(), 2),
        "Sharpe Ratio": SHARPE_CAVEAT,
    }
    return trades, summary


def run_baseline_no_stoploss(events: pd.DataFrame, prices: dict,
                             calendar: pd.DatetimeIndex) -> tuple:
    """對照組：沿用 baseline 進場設定但完全不停損，純持有到 period_end。

    進場設定與 BASELINE_CONFIG 相同（entry_day_index=5、開盤價進場），
    成本參數不變，唯獨關閉停損——以 stop_loss_pct=1.0 使停損線為 0、
    永不觸發，等同不設停損、抱到 period_end 收盤。不修改 BASELINE_CONFIG
    或 run_baseline()，這是獨立的比較函式。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。

    Returns:
        (trades, summary) — 交易明細 DataFrame 與績效摘要 dict。
    """
    cfg = BASELINE_CONFIG
    all_trades = build_trade_level(
        events, prices, calendar,
        slippage_pct=cfg["slippage_pct"],
        stop_loss_pct=None,                # None = 完全不套用停損
        stop_fill_mode=cfg["stop_fill_mode"],
        entry_price_mode=cfg["entry_price_mode"],
        stop_from_entry_day=cfg["stop_from_entry_day"],
        verbose=False,
    )
    trades = all_trades[
        all_trades["entry_day_index"] == cfg["entry_day_index"]
    ].copy()

    ret = trades["return_pct"]
    n_stop = int((trades["exit_reason"] == "stop_loss").sum())
    summary = {
        "樣本數": len(trades),
        "勝率%": round((ret > 0).mean() * 100, 1),
        "停損觸發次數（應為0）": n_stop,
        "平均報酬率%": round(ret.mean() * 100, 2),
        "中位數報酬率%": round(ret.median() * 100, 2),
        "報酬率標準差%": round(ret.std() * 100, 2),
        "平均pnl_ntd": round(trades["pnl_ntd"].mean()),
        "總pnl_ntd": round(trades["pnl_ntd"].sum()),
        "平均持有天數": round(trades["holding_days"].mean(), 2),
    }
    return trades, summary


# 三組名稱（皆可執行、無 look-ahead bias）。第一組不再依「未來是否升級」
# 篩選，因為下單當下無法得知這次第一次處置未來會不會升級。
GROUP_FIRST = "第一次處置"        # 全部第一次（不分未來是否升級）
GROUP_ESCALATED = "接續升級20分"   # 第二次以上，緊接自己前一次第一次（重疊）
GROUP_INDEPENDENT = "獨立20分"     # 第二次以上，獨立發生
GROUP_ORDER = [GROUP_FIRST, GROUP_ESCALATED, GROUP_INDEPENDENT]


def assign_escalation_groups(events: pd.DataFrame,
                             calendar: pd.DatetimeIndex) -> pd.Series:
    """依 disposition_order 將事件分為三組（皆可執行、無 look-ahead）。

    分組（每筆恰屬一組，涵蓋全部事件）：
      第一次處置   —— disposition_order == 第一次（**全部**，不論未來是否升級；
                     下單當下無法得知未來升級與否，故不以此篩選）
      接續升級20分 —— 第二次以上，且緊接自己前一次第一次處置（嚴格重疊：
                     本次 period_start 落在前次第一次的 [period_start, period_end]）
      獨立20分     —— 第二次以上，且非接續前次第一次而來

    第二、三組的區分只用歷史資訊（本次事件發生時，前次處置早已公告），
    故可執行；第一組不做「是否升級」的未來篩選，同樣可執行。

    Args:
        events: 處置事件 DataFrame（需含 stock_id/period_start/period_end/
            disposition_order）。
        calendar: 交易日曆（保留參數以利未來擴充，本函式未直接使用）。

    Returns:
        與 events 對齊（同 index）的分組 Series。
    """
    ev = events.copy()
    ev["_esc_second"] = False
    order = ev.sort_values(["stock_id", "period_start"])
    for _, g in order.groupby("stock_id"):
        idx = g.index.tolist()
        for ia, ib in zip(idx[:-1], idx[1:]):
            a, b = ev.loc[ia], ev.loc[ib]
            if (a["disposition_order"] == "第一次"
                    and b["disposition_order"] == "第二次以上"
                    and a["period_start"] <= b["period_start"] <= a["period_end"]):
                ev.at[ib, "_esc_second"] = True

    def _grp(r):
        if r["disposition_order"] == "第一次":
            return GROUP_FIRST
        return GROUP_ESCALATED if r["_esc_second"] else GROUP_INDEPENDENT

    return ev.apply(_grp, axis=1)


def three_group_performance(events: pd.DataFrame, prices: dict,
                            calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """以現行 baseline（無停損）計算三組（第一次／接續升級／獨立20分）績效。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。

    Returns:
        每組一列，含 group/n/win/mean/median/mean_pnl/se 的 DataFrame。
    """
    ev = events.copy()
    ev["group"] = assign_escalation_groups(ev, calendar)
    trades, _ = run_baseline(ev, prices, calendar)
    trades = trades.merge(ev[["stock_id", "period_start", "group"]],
                          on=["stock_id", "period_start"], how="left")

    rows = []
    for grp in GROUP_ORDER:
        g = trades[trades["group"] == grp]
        ret = g["return_pct"]
        rows.append({
            "group": grp,
            "n": len(g),
            "win": round((ret > 0).mean() * 100, 1),
            "mean": round(ret.mean() * 100, 2),
            "median": round(ret.median() * 100, 2),
            "mean_pnl": round(g["pnl_ntd"].mean()),
            "se": ret.std() / len(g) ** 0.5 * 100,
        })
    return pd.DataFrame(rows)


def compare_entry_price_mode(events: pd.DataFrame, prices: dict,
                             calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """比較第 4 天以開盤價 vs 收盤價進場的績效。

    其餘設定一律沿用 BASELINE_CONFIG。開盤價進場另外並列「進場當日
    起算停損」的版本，因為開盤建倉後當日盤中風險為真。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。

    Returns:
        並排比較表。
    """
    cfg = BASELINE_CONFIG
    variants = [
        ("收盤價進場（baseline）", "close", False),
        ("開盤價進場（停損自隔日起）", "open", False),
        ("開盤價進場（停損自當日起）", "open", True),
    ]

    rows = []
    for label, price_mode, stop_from_entry in variants:
        trades = build_trade_level(
            events, prices, calendar,
            slippage_pct=cfg["slippage_pct"],
            stop_loss_pct=cfg["stop_loss_pct"],
            stop_fill_mode=cfg["stop_fill_mode"],
            entry_price_mode=price_mode,
            stop_from_entry_day=stop_from_entry,
            verbose=False,
        )
        trades = trades[trades["entry_day_index"] == cfg["entry_day_index"]]
        ret = trades["return_pct"]
        stopped = trades["exit_reason"] == "stop_loss"
        rows.append({
            "版本": label,
            "樣本數": len(trades),
            "勝率%": round((ret > 0).mean() * 100, 1),
            "停損觸發率%": round(stopped.mean() * 100, 1),
            "平均%": round(ret.mean() * 100, 2),
            "中位數%": round(ret.median() * 100, 2),
            "標準差%": round(ret.std() * 100, 2),
            "平均pnl_ntd": round(trades["pnl_ntd"].mean()),
            "總pnl_ntd": round(trades["pnl_ntd"].sum()),
            "平均持有天數": round(trades["holding_days"].mean(), 2),
        })
    return pd.DataFrame(rows)


def compare_stop_basis(events: pd.DataFrame, prices: dict,
                       calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """比較兩種停損線基準：每日重算 vs 進場價固定。

    除 stop_basis 外，所有參數一律沿用 BASELINE_CONFIG（第 5 天開盤價進場、
    進場當日即檢查停損、9% 停損、其餘成本不變），以隔離停損基準的效果。

    Args:
        events: 處置事件 DataFrame。
        prices: load_prices 產生的日K字典。
        calendar: 交易日曆。

    Returns:
        並排比較表。
    """
    cfg = BASELINE_CONFIG
    variants = [
        ("trailing_daily（每日重算，baseline）", "trailing_daily"),
        ("entry_fixed（進場價固定）", "entry_fixed"),
    ]

    rows = []
    for label, basis in variants:
        trades = build_trade_level(
            events, prices, calendar,
            slippage_pct=cfg["slippage_pct"],
            stop_loss_pct=cfg["stop_loss_pct"],
            stop_fill_mode=cfg["stop_fill_mode"],
            entry_price_mode=cfg["entry_price_mode"],
            stop_from_entry_day=cfg["stop_from_entry_day"],
            stop_basis=basis,
            verbose=False,
        )
        trades = trades[trades["entry_day_index"] == cfg["entry_day_index"]]
        ret = trades["return_pct"]
        stopped = trades["exit_reason"] == "stop_loss"
        rows.append({
            "停損基準": label,
            "樣本數": len(trades),
            "勝率%": round((ret > 0).mean() * 100, 1),
            "停損觸發率%": round(stopped.mean() * 100, 1),
            "平均%": round(ret.mean() * 100, 2),
            "中位數%": round(ret.median() * 100, 2),
            "平均持有天數": round(trades["holding_days"].mean(), 2),
            "平均pnl_ntd": round(trades["pnl_ntd"].mean()),
            "總pnl_ntd": round(trades["pnl_ntd"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """執行事件回測與停損模擬。"""
    load_dotenv()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    events = pd.read_csv(CLEAN_CSV, dtype={"stock_id": str})
    for col in ["period_start", "period_end"]:
        events[col] = pd.to_datetime(events[col])

    _section("步驟1：抓取股價資料")
    print(f"處置事件 {len(events)} 筆，涉及 {events['stock_id'].nunique()} 檔股票")
    calendar = fetch_trading_calendar("2019-12-01", "2026-08-31")
    ranges = build_price_ranges(events, calendar)
    prices = load_prices(ranges)
    print(f"成功載入 {len(prices)} 檔日K")

    _section("成本與停損模型")
    print(f"手續費率 {FEE_RATE} x 折扣 {FEE_DISCOUNT} = "
          f"單邊 {EFFECTIVE_FEE:.6f}（買賣雙邊各收一次）")
    print(f"證交稅率 {TAX_RATE}（僅出場收一次）｜滑價 {SLIPPAGE_PCT:.1%}")
    print(f"停損 {STOP_LOSS_PCT:.0%}，停損線每日重算 = 前一日收盤 x "
          f"{1 - STOP_LOSS_PCT:.2f}")
    print("觸發判斷：當日 low <= 當日停損線")

    # --- BASELINE：目前選定的正式版本 ---------------------------------
    _section("BASELINE —— 目前選定的正式版本")
    baseline_trades, baseline = run_baseline(events, prices, calendar)

    print("設定：")
    for key, value in BASELINE_CONFIG.items():
        print(f"  {key:<18} = {value}")
    print()
    print("績效摘要：")
    for key, value in baseline.items():
        if key == "Sharpe Ratio":
            continue
        if isinstance(value, (int,)) and abs(value) >= 1000:
            print(f"  {key:<16} : {value:>14,}")
        else:
            print(f"  {key:<16} : {value:>14}")
    print()
    print(f"  {'Sharpe Ratio':<16} : {SHARPE_CAVEAT}")

    pd.DataFrame(
        [{"項目": k, "數值": v} for k, v in
         list(BASELINE_CONFIG.items()) + list(baseline.items())]
    ).to_csv(BASELINE_CSV, index=False)
    print()
    print(f"[輸出] {BASELINE_CSV}")

    _section("進場價假設：開盤價 vs 收盤價（entry_day_index=4）")
    print(compare_entry_price_mode(events, prices, calendar).to_string(index=False))

    _section("停損基準：每日重算 vs 進場價固定"
             f"（entry_day_index={BASELINE_CONFIG['entry_day_index']}、開盤進場）")
    print(compare_stop_basis(events, prices, calendar).to_string(index=False))

    _section(f"以下為掃描／敏感度測試（非 baseline）"
             f"｜基準情境（動態 {STOP_LOSS_PCT:.0%} 停損、low 觸發、"
             f"停損成交=stop_line）")
    trades = build_trade_level(events, prices, calendar,
                               stop_fill_mode="stop_line")
    trades.to_csv(TRADE_CSV, index=False)
    print(f"[輸出] {TRADE_CSV}（{len(trades)} 筆）")

    _section("步驟3：滑價敏感度測試（3 種滑價 x 2 種停損成交模式）")
    rows = []
    for mode in STOP_FILL_MODES:
        base = build_trade_level(events, prices, calendar,
                                 stop_fill_mode=mode, verbose=False)
        for slip in SLIPPAGE_SCENARIOS:
            stats = {"停損成交模式": mode, "滑價%": f"{slip:.1%}"}
            stats.update(scenario_stats(apply_slippage(base, slip)))
            rows.append(stats)
    matrix = pd.DataFrame(rows).sort_values(["滑價%", "停損成交模式"])
    print(matrix.to_string(index=False))

    _section("三種滑價情境的整體 Sharpe（合併全部 entry_day_index）")
    sharpe_table = matrix.pivot(index="滑價%", columns="停損成交模式",
                                values="Sharpe")
    print(sharpe_table.to_string())

    _section("損益兩平檢查")
    for _, row in matrix.iterrows():
        verdict = "獲利" if row["平均%"] > 0 else "虧損"
        print(f"  滑價 {row['滑價%']:>5} / {row['停損成交模式']:>12}："
              f"平均 {row['平均%']:>6.2f}%  Sharpe {row['Sharpe']:>6.3f}"
              f"  -> {verdict}")

    _section("跳空跌破停損線的比例（stop_line 模式高估的來源）")
    stopped = trades[trades["exit_reason"] == "stop_loss"]
    gapped = stopped["gapped"]
    print(f"停損出場 {len(stopped)} 筆，其中開盤已跳空跌破停損線 "
          f"{int(gapped.sum())} 筆（{gapped.mean() * 100:.1f}%）")

    _section("依 entry_day_index 分組（基準情境）")
    print(summarize(trades, "entry_day_index").to_string())

    _section("停損觸發率隨 entry_day_index 的變化")
    rate = trades.groupby("entry_day_index")["exit_reason"].agg(
        樣本數="size",
        停損次數=lambda s: (s == "stop_loss").sum(),
    )
    rate["停損率%"] = (rate["停損次數"] / rate["樣本數"] * 100).round(1)
    rate["平均持有天數"] = trades.groupby("entry_day_index")[
        "holding_days"].mean().round(2)
    print(rate.to_string())


if __name__ == "__main__":
    main()
