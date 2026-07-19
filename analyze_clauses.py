"""處置事件款號組合的做多績效分析（現行 baseline：第 2 天、開盤、無停損）。

標記每筆處置事件在回溯窗口內觸發的注意款號（第 1~8 款），輸出
data/disposition_events_with_clauses.csv，並以 baseline 計算：
  (a) 單一款號邊際表現（允許重疊）；
  (b) 實際款號組合的表現（含小樣本標註）。

無 look-ahead：款號由事件公告時已存在的歷史注意股紀錄反推，屬事件當下
即可得知的資訊。

用法：
    python analyze_clauses.py
"""

import pandas as pd
from dotenv import load_dotenv

from data_loader import fetch_trading_calendar
from event_backtest import (
    CLEAN_CSV, CLAUSE_CSV, GROUP_ORDER, build_clause_dataset,
    build_price_ranges, load_prices, run_baseline,
)

SMALL_N = 30   # 樣本數低於此值標註「僅供參考」


def _section(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    load_dotenv()
    pd.set_option("display.width", 200)

    # 1. 標記款號並輸出資料集（可重複執行）
    _section("步驟1：標記款號 → " + CLAUSE_CSV)
    tagged = build_clause_dataset(CLEAN_CSV, CLAUSE_CSV)
    print("condition_category 分布：",
          tagged["condition_category"].value_counts().to_dict())
    print(f"查無對應注意紀錄（缺失）: {int(tagged['clause_missing'].sum())} 筆")

    # 2. baseline 交易（第 2 天、開盤、無停損），掛上款號欄
    events = pd.read_csv(CLEAN_CSV, dtype={"stock_id": str})
    for col in ["period_start", "period_end"]:
        events[col] = pd.to_datetime(events[col])
    calendar = fetch_trading_calendar("2019-12-01", "2026-08-31")
    prices = load_prices(build_price_ranges(events, calendar))
    trades, _ = run_baseline(events, prices, calendar)

    key = ["stock_id", "period_start"]
    tagged["period_start"] = pd.to_datetime(tagged["period_start"])
    clause_cols = [f"clause_{c}" for c in range(1, 9)] + \
        ["clause_set", "clause_missing"]
    trades = trades.merge(tagged[key + clause_cols], on=key, how="left")

    # (a) 單一款號邊際表現（允許重疊）
    _section("步驟2a：單一款號邊際表現（第 1~8 款，允許重疊）")
    print(f"{'款號':<8}{'樣本數':>8}{'勝率%':>8}{'平均%':>9}"
          f"{'中位數%':>9}{'平均pnl_ntd':>13}")
    for c in range(1, 9):
        g = trades[trades[f"clause_{c}"] == True]  # noqa: E712
        if len(g) == 0:
            print(f"第{c}款{'':<4}{0:>8}{'-':>8}{'-':>9}{'-':>9}{'-':>13}")
            continue
        ret = g["return_pct"]
        note = "（含連續三日）" if c == 1 else ""
        print(f"{'第'+str(c)+'款'+note:<8}{len(g):>8}{(ret>0).mean()*100:>8.1f}"
              f"{ret.mean()*100:>9.2f}{ret.median()*100:>9.2f}"
              f"{g['pnl_ntd'].mean():>13,.0f}")

    # (b) 實際組合表現
    _section("步驟2b：實際款號組合表現")
    valid = trades[~trades["clause_missing"].fillna(True) &
                   (trades["clause_set"].fillna("") != "")].copy()
    combo = valid.groupby("clause_set")["return_pct"].agg(
        n="size", mean=lambda s: round(s.mean() * 100, 2),
        median=lambda s: round(s.median() * 100, 2))
    combo["mean_pnl"] = valid.groupby("clause_set")["pnl_ntd"].mean().round()
    combo["total_pnl"] = valid.groupby("clause_set")["pnl_ntd"].sum().round()
    combo = combo.reset_index()

    print(f"\n出現最多的前 15 種組合（按樣本數）：")
    top = combo.sort_values("n", ascending=False).head(15)
    _print_combo(top)

    print(f"\n組合報酬排行榜（樣本數 >= {SMALL_N}，按平均報酬率排序）：")
    big = combo[combo["n"] >= SMALL_N].sort_values("mean", ascending=False)
    _print_combo(big)

    print(f"\n小樣本組合（n < {SMALL_N}，數字僅供參考、不列入主要結論）：")
    small = combo[combo["n"] < SMALL_N].sort_values("mean", ascending=False)
    print(f"  共 {len(small)} 種小樣本組合；前 5 高平均：")
    _print_combo(small.head(5), indent="  ")


def _print_combo(df: pd.DataFrame, indent: str = "") -> None:
    print(f"{indent}{'組合':<16}{'樣本數':>7}{'平均%':>9}"
          f"{'中位數%':>9}{'平均pnl':>12}{'總pnl':>14}")
    for _, r in df.iterrows():
        flag = " *小樣本" if r["n"] < SMALL_N else ""
        print(f"{indent}({r['clause_set']}){'':<{max(0,14-len(r['clause_set']))}}"
              f"{int(r['n']):>7}{r['mean']:>9.2f}{r['median']:>9.2f}"
              f"{r['mean_pnl']:>12,.0f}{r['total_pnl']:>14,.0f}{flag}")


if __name__ == "__main__":
    main()
