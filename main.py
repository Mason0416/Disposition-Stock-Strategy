"""主程式：示範完整投資組合當沖回測流程。

從 .env 讀取 FinMind token，載入一組股票池、產生訊號矩陣、
執行投資組合回測，最後輸出績效報告並繪製資金曲線與曝險圖。
"""

import pandas as pd
from dotenv import load_dotenv

from data_loader import load_multiple
from portfolio_backtest import PortfolioBacktester
from signals import generate_signals_matrix


# 範例股票池：台積電、鴻海、聯發科、台達電、元大台灣50
STOCK_POOL = ["2330", "2317", "2454", "2308", "0050"]
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


def build_example_signals(data: dict) -> pd.DataFrame:
    """產生範例訊號矩陣。

    此處直接呼叫 signals.py 的 generate_signals_matrix。
    預設該函數回傳全 0 矩陣（不交易）——請編輯 signals.py
    填入你自己的策略邏輯。只要格式符合（index 為日期、
    columns 為股票代號、值為 1/-1/0）即可直接使用。

    Args:
        data: dict[str, pd.DataFrame]，多股票 OHLCV 字典。

    Returns:
        訊號矩陣 DataFrame。
    """
    return generate_signals_matrix(data)


def main() -> None:
    """執行投資組合回測範例流程。"""
    # 讀取 .env 中的 FINMIND_TOKEN（供 data_loader 內部使用）
    load_dotenv()

    # 1. 載入股票池資料
    data = load_multiple(STOCK_POOL, START_DATE, END_DATE)
    print(f"已載入 {len(data)} 檔股票：{list(data.keys())}")

    # 2. 產生訊號矩陣
    #    使用者應自行替換為自己策略產生的矩陣（格式：index=日期、
    #    columns=股票代號、值為 1/-1/0）。此處用範例均線交叉策略示範。
    signals = build_example_signals(data)
    print(f"訊號矩陣 shape：{signals.shape}，"
          f"買入 {int((signals == 1).sum().sum())} 次，"
          f"賣出 {int((signals == -1).sum().sum())} 次")

    # 3. 執行投資組合回測
    backtester = PortfolioBacktester()
    trades = backtester.run(data, signals)
    print(f"完成回測，共 {len(trades)} 筆交易\n")

    if trades.empty:
        print("沒有交易訊號，略過績效報告與資金曲線。")
        return

    # 4. 輸出報告
    backtester.report()

    # 5. 繪製資金曲線與曝險圖
    backtester.plot_equity_curve()


if __name__ == "__main__":
    main()
