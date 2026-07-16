"""處置股資料抓取驗證腳本。

純資料檢查：抓取處置股名單與上市櫃對照表後，印出欄位格式、
欄位值分布與抽樣內容，供人工確認 FinMind 實際回傳格式。
不包含任何策略或統計分析邏輯。

用法：
    python verify_disposition_data.py
"""

import datetime as dt

import pandas as pd
from dotenv import load_dotenv

from data_loader import fetch_disposition_events, fetch_stock_market_type

START_DATE = "2020-01-01"
RANDOM_SEED = 42


def _section(title: str) -> None:
    """印出區段標題分隔線。"""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    """執行資料抓取並印出各項欄位檢查結果。"""
    # 讀取 .env 中的 FINMIND_TOKEN（供 data_loader 內部使用）
    load_dotenv()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)

    end_date = dt.date.today().strftime("%Y-%m-%d")

    _section("抓取處置股名單 TaiwanStockDispositionSecuritiesPeriod")
    events = fetch_disposition_events(START_DATE, end_date)

    _section("0. 實際回傳欄位與前 20 筆內容")
    print("columns:", list(events.columns))
    print()
    print("dtypes:")
    print(events.dtypes)
    print()
    print(events.head(20).to_string())

    _section("1. 總筆數與日期範圍")
    print(f"總筆數: {len(events)}")
    print(f"最早日期: {events['date'].min()}")
    print(f"最晚日期: {events['date'].max()}")
    print(f"不重複股票檔數: {events['stock_id'].nunique()}")

    _section("2. condition 欄位所有不重複值")
    if "condition" in events.columns:
        conditions = events["condition"].dropna().unique()
        print(f"共 {len(conditions)} 種不重複值：")
        print()
        for value in sorted(conditions, key=str):
            print(f"  - {value!r}")
    else:
        print("[警告] 回傳資料中無 condition 欄位")

    _section("3. disposition_cnt 數值分布")
    if "disposition_cnt" in events.columns:
        counts = events["disposition_cnt"].value_counts(dropna=False)
        print(counts.sort_index().to_string())
    else:
        print("[警告] 回傳資料中無 disposition_cnt 欄位")

    _section("4. measure 欄位 5 筆完整內容")
    if "measure" in events.columns:
        for i, value in enumerate(events["measure"].dropna().head(5), 1):
            print(f"[{i}] {value!r}")
            print()
    else:
        print("[警告] 回傳資料中無 measure 欄位")

    _section("5. 隨機抽 5 檔股票（核對 period_start ~ period_end 區間長度）")
    wanted = ["stock_id", "stock_name", "disposition_cnt", "condition",
              "period_start", "period_end"]
    available = [c for c in wanted if c in events.columns]
    sample_n = min(5, len(events))
    if sample_n:
        sample = events.sample(n=sample_n, random_state=RANDOM_SEED)
        print(sample[available].to_string(index=False))
    else:
        print("[警告] 無資料可抽樣")

    _section("抓取上市櫃對照表 TaiwanStockInfo")
    info = fetch_stock_market_type()

    _section("6. TaiwanStockInfo type 欄位不重複值與數量")
    print("columns:", list(info.columns))
    print(f"總筆數: {len(info)}")
    print()
    if "type" in info.columns:
        print(info["type"].value_counts(dropna=False).to_string())
    else:
        print("[警告] 回傳資料中無 type 欄位")


if __name__ == "__main__":
    main()
