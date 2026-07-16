"""處置事件資料清理／正規化腳本。

讀取 data/disposition_events.csv 與 data/stock_info.csv，逐層過濾成
「一般股票、標準 5 分／20 分撮合規格」的乾淨資料，輸出
data/disposition_events_clean.csv。

純資料清理：不含任何進出場規則、訊號或回測邏輯。

用法：
    python clean_disposition_data.py
"""

import re

import pandas as pd
from dotenv import load_dotenv

from data_loader import fetch_trading_calendar

# condition 原始值 -> 正規化類別（精確對照，不做模糊字串包含比對）
CONDITION_MAP = {
    # 連續三日（含當沖加嚴變體）
    "連續三次": "連續三日",
    "連續3個營業日": "連續三日",
    "連續三次及當日沖銷標準": "連續三日",
    "連續3個營業日及沖銷標準": "連續三日",
    "因連續3個營業日達本中心作業要點第四條第一項第一款": "連續三日",
    # 連續五日（含當沖加嚴變體）
    "連續五次": "連續五日",
    "連續5個營業日": "連續五日",
    "連續五次及當日沖銷標準": "連續五日",
    "連續5個營業日及沖銷標準": "連續五日",
    # 十日六次
    "最近十個營業日已有六次": "十日六次",
    "最近10個營業日內有6個營業日": "十日六次",
    # 三十日十二次
    "最近三十個營業日已有十二次": "三十日十二次",
    "最近30個營業日內有12個營業日": "三十日十二次",
    # 其他
    "最近6個營業日內有4個營業日": "六日四次_待確認",
    "監視業務督導會報決議": "督導會報決議",
    "轉(交)換公司債之標的證券經本中心或臺灣證券交易所發布處置": "可轉債連動處置",
}

# 一般股票的標準撮合規格（分鐘）-> 處置次序
STANDARD_INTERVAL_ORDER = {
    5: "第一次",
    20: "第二次以上",
}

# 非一般股票規格的撮合分鐘數（變更交易方法 10/25、分盤集合競價 45/60）
NON_STANDARD_INTERVALS = {10, 25, 45, 60}

# TWSE 短碼 measure -> 處置次序
TWSE_ORDER_MAP = {
    "第一次處置": "第一次",
    "第二次處置": "第二次以上",
}

# 非一般股票的 industry_category（DR／ETF／ETN）。
# 這些屬性不隨時間改變，故以「任一列命中即排除」判斷。
NON_COMMON_STOCK_CATEGORIES = {
    "存託憑證",
    "ETF",
    "上櫃ETF",
    "上櫃指數股票型基金(ETF)",
    "指數投資證券(ETN)",
}

# 創新板股票簡稱後綴（「-創」／「-KY創」）。
# 創新板身分會隨轉板改變，而 stock_info 的 industry_category 是跨年份累積的
# 多對多標籤，無法還原事件當下狀態；處置資料的 stock_name 為事件當下名稱，
# 故以此判斷。錨定字尾以避免誤傷緯創／群創／鈺創等一般公司。
INNOVATION_BOARD_PATTERN = r"-(?:KY)?創$"

# 標準處置期間長度（交易日）。10 = 一般、12 = 當沖加嚴。
# 其餘值來自颱風休市使公告 period_end 未反映順延。
STANDARD_TRADING_DAYS = (10, 12)

RAW_CSV = "data/disposition_events.csv"
INFO_CSV = "data/stock_info.csv"
OUT_CSV = "data/disposition_events_clean.csv"


def _section(title: str) -> None:
    """印出區段標題分隔線。"""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def normalize_condition(df: pd.DataFrame) -> pd.DataFrame:
    """依 CONDITION_MAP 正規化 condition 欄位，未列入者標記為「未分類」。

    Args:
        df: 含 condition 欄位的 DataFrame。

    Returns:
        新增 condition_category 欄位的 DataFrame。
    """
    df = df.copy()
    df["condition_category"] = df["condition"].map(CONDITION_MAP)

    unmapped = df[df["condition_category"].isna()]
    if not unmapped.empty:
        print("[警告] 以下 condition 原始值不在對照表中，歸類為「未分類」：")
        for value, cnt in unmapped["condition"].value_counts().items():
            print(f"  - {value!r}  ({cnt} 筆)")
    df["condition_category"] = df["condition_category"].fillna("未分類")
    return df


def parse_measure(df: pd.DataFrame) -> pd.DataFrame:
    """解析 measure，產生 matching_interval_minutes 與 disposition_order。

    TWSE 以短碼文字（第一次處置／第二次處置）判斷次序；
    TPEx 以長文中的撮合分鐘數查表判斷（5 -> 第一次，20 -> 第二次以上）。
    兩者皆無法判斷者標記為「未知」。

    Args:
        df: 含 measure 欄位的 DataFrame。

    Returns:
        新增 matching_interval_minutes 與 disposition_order 欄位的 DataFrame。
    """
    df = df.copy()
    measure = df["measure"].fillna("")

    minutes = measure.str.extract(r"約每\s*(\d+)\s*分鐘撮合")[0]
    df["matching_interval_minutes"] = pd.to_numeric(minutes, errors="coerce")

    order_text = measure.str.extract(r"^(第[一二]次處置)$")[0]

    # TWSE 短碼優先；其次以 TPEx 分鐘數查表
    from_text = order_text.map(TWSE_ORDER_MAP)
    from_minutes = df["matching_interval_minutes"].map(STANDARD_INTERVAL_ORDER)
    df["disposition_order"] = from_text.fillna(from_minutes).fillna("未知")
    return df


def add_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """以真實交易日曆計算 period_start ~ period_end 的交易日數（含頭尾）。

    Args:
        df: 含 period_start / period_end 欄位的 DataFrame。

    Returns:
        新增 trading_days 欄位的 DataFrame。
    """
    df = df.copy()
    starts = pd.to_datetime(df["period_start"])
    ends = pd.to_datetime(df["period_end"])

    cal_start = min(starts.min(), ends.min()).strftime("%Y-%m-%d")
    cal_end = max(starts.max(), ends.max()).strftime("%Y-%m-%d")
    calendar = fetch_trading_calendar(cal_start, cal_end)

    cal_values = calendar.values
    # searchsorted 計算兩端點之間的交易日數（含頭尾）
    left = cal_values.searchsorted(starts.values, side="left")
    right = cal_values.searchsorted(ends.values, side="right")
    df["trading_days"] = right - left
    return df


def main() -> None:
    """執行逐層過濾並輸出乾淨資料。"""
    load_dotenv()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    funnel = []

    raw = pd.read_csv(RAW_CSV, dtype={"stock_id": str})
    raw["date"] = pd.to_datetime(raw["date"])
    info = pd.read_csv(INFO_CSV, dtype={"stock_id": str})
    funnel.append(("L0 原始資料", len(raw), ""))

    # --- L1：join stock_info，只留 twse/tpex -----------------------------
    _section("L1：stock_info join，只留 type 為 twse/tpex")
    info_unique = info[["stock_id", "type"]].drop_duplicates("stock_id")
    df = raw.merge(info_unique, on="stock_id", how="left")
    df = df.rename(columns={"type": "market"})

    before = len(df)
    keep_mask = df["market"].isin(["twse", "tpex"])
    dropped_l1 = df[~keep_mask].copy()
    df = df[keep_mask].copy()

    print(f"過濾前 {before} 筆 -> 過濾後 {len(df)} 筆"
          f"（排除 {len(dropped_l1)} 筆）")
    print()
    print("排除原因統計：")
    reason = dropped_l1["market"].fillna("join 不到（權證/可轉債/DR 等）")
    print(reason.value_counts().to_string())
    print()
    print("被排除標的隨機抽 10 筆：")
    sample_n = min(10, len(dropped_l1))
    if sample_n:
        print(dropped_l1.sample(n=sample_n, random_state=42)[
            ["stock_id", "stock_name", "market"]].to_string(index=False))
    funnel.append(("L1 只留 twse/tpex", len(df), f"-{len(dropped_l1)}"))

    # --- L2：condition 正規化 + measure 解析 -----------------------------
    _section("L2：condition 正規化與 disposition_order 判斷（不過濾）")
    df = normalize_condition(df)
    df = parse_measure(df)
    print("condition_category 分布：")
    print(df["condition_category"].value_counts().to_string())
    print()
    print("disposition_order 分布：")
    print(df["disposition_order"].value_counts().to_string())
    print()
    print("matching_interval_minutes 分布（NaN = TWSE 短碼無分鐘數）：")
    print(df["matching_interval_minutes"].value_counts(dropna=False).to_string())
    funnel.append(("L2 正規化（未過濾）", len(df), "0"))

    # --- L3：只留一般股票標準 5 分／20 分規格 ----------------------------
    _section("L3：只留一般股票標準 5 分／20 分撮合規格")
    before = len(df)
    minutes = df["matching_interval_minutes"]

    is_non_standard = minutes.isin(NON_STANDARD_INTERVALS)
    is_unknown_order = df["disposition_order"] == "未知"
    drop_mask = is_non_standard | is_unknown_order

    dropped_l3 = df[drop_mask].copy()
    df = df[~drop_mask].copy()

    print(f"過濾前 {before} 筆 -> 過濾後 {len(df)} 筆"
          f"（排除 {len(dropped_l3)} 筆）")
    print()
    print("排除原因統計：")
    reasons = []
    for _, row in dropped_l3.iterrows():
        if row["matching_interval_minutes"] in NON_STANDARD_INTERVALS:
            mins = int(row["matching_interval_minutes"])
            label = ("變更交易方法" if mins in (10, 25) else "分盤集合競價")
            reasons.append(f"非標準規格 {mins} 分（{label}）")
        else:
            reasons.append(f"無次序資訊（measure={row['measure']}）")
    print(pd.Series(reasons).value_counts().to_string())
    funnel.append(("L3 只留標準 5/20 分", len(df), f"-{len(dropped_l3)}"))

    # --- trading_days ---------------------------------------------------
    _section("計算 trading_days（真實交易日曆）")
    df = add_trading_days(df)
    print("trading_days 分布：")
    print(df["trading_days"].value_counts().sort_index().to_string())

    # --- L4：排除非一般股票（DR／ETF／ETN／創新板）------------------------
    _section("L4：排除非一般股票（DR／ETF／ETN／創新板）")
    before = len(df)

    # DR／ETF／ETN：屬性不隨時間變動，任一列命中即排除
    non_common_ids = set(
        info.loc[
            info["industry_category"].isin(NON_COMMON_STOCK_CATEGORIES),
            "stock_id",
        ]
    )
    is_non_common = df["stock_id"].isin(non_common_ids)
    # 創新板：以事件當下名稱後綴判斷
    is_innovation = df["stock_name"].str.contains(
        INNOVATION_BOARD_PATTERN, regex=True, na=False
    )

    drop_mask = is_non_common | is_innovation
    dropped_l4 = df[drop_mask].copy()
    df = df[~drop_mask].copy()

    print(f"過濾前 {before} 筆 -> 過濾後 {len(df)} 筆"
          f"（排除 {len(dropped_l4)} 筆）")
    print()
    print("排除原因統計：")
    cat_by_id = (
        info[info["industry_category"].isin(NON_COMMON_STOCK_CATEGORIES)]
        .groupby("stock_id")["industry_category"]
        .apply(lambda s: "／".join(sorted(set(s))))
    )
    reasons = dropped_l4["stock_id"].map(cat_by_id)
    reasons = reasons.fillna("創新板股票（依事件當下名稱後綴）")
    print(reasons.value_counts().to_string())
    print()
    print("被排除標的（依檔數）：")
    print(dropped_l4.groupby(["stock_id", "stock_name"]).size()
          .rename("筆數").reset_index().to_string(index=False))
    funnel.append(("L4 排除非一般股票", len(df), f"-{len(dropped_l4)}"))

    # --- L5：只留 trading_days 為 10 或 12 -------------------------------
    _section("L5：只留 trading_days 為 10 或 12 的標準規格")
    before = len(df)
    keep_mask = df["trading_days"].isin(STANDARD_TRADING_DAYS)
    dropped_l5 = df[keep_mask == False].copy()  # noqa: E712
    df = df[keep_mask].copy()

    print(f"過濾前 {before} 筆 -> 過濾後 {len(df)} 筆"
          f"（排除 {len(dropped_l5)} 筆）")
    print()
    print("排除原因統計（trading_days 異常值，肇因於颱風休市未反映順延）：")
    print(dropped_l5["trading_days"].value_counts().sort_index()
          .rename("筆數").to_string())
    print()
    print("被排除列的年份分布：")
    print(pd.to_datetime(dropped_l5["date"]).dt.year.value_counts()
          .sort_index().to_string())
    funnel.append(("L5 只留 10/12 交易日", len(df), f"-{len(dropped_l5)}"))

    # --- 輸出 -----------------------------------------------------------
    out_cols = list(raw.columns) + [
        "market", "condition_category", "matching_interval_minutes",
        "disposition_order", "trading_days",
    ]
    df = df[out_cols].sort_values(["date", "stock_id"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    _section("過濾漏斗")
    funnel_df = pd.DataFrame(funnel, columns=["步驟", "剩餘筆數", "本步驟增減"])
    funnel_df["佔原始比例"] = (
        funnel_df["剩餘筆數"] / len(raw) * 100
    ).round(1).astype(str) + "%"
    print(funnel_df.to_string(index=False))

    _section("最終 trading_days 分布")
    print(df["trading_days"].value_counts().sort_index().to_string())

    _section("最終交叉統計：disposition_order x condition_category")
    print(pd.crosstab(
        df["condition_category"],
        df["disposition_order"],
        margins=True,
        margins_name="合計",
    ).to_string())

    _section("最終交叉統計：market x condition_category x disposition_order")
    cross = pd.crosstab(
        [df["market"], df["condition_category"]],
        df["disposition_order"],
        margins=True,
        margins_name="合計",
    )
    print(cross.to_string())

    print()
    print(f"[輸出] {OUT_CSV}（{len(df)} 筆，{len(df.columns)} 欄）")


if __name__ == "__main__":
    main()
