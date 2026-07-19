"""資料載入模組。

負責從 FinMind 抓取台股日線資料，或從本地 CSV 讀取，
並統一輸出標準 OHLCV DataFrame。
另提供處置股名單、上市/上櫃對照表、注意股名單的抓取與快取。
"""

import json
import os
import re
import time

import pandas as pd
import requests
from FinMind.data import DataLoader


# FinMind 原始欄位 -> 標準欄位 對應
_COLUMN_MAP = {
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
}

# 標準 OHLCV 欄位（需確保為 float）
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _to_standard_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """將任意來源的 DataFrame 轉為標準 OHLCV 格式。

    確保 index 為 DatetimeIndex，欄位為 open/high/low/close/volume，
    且數值型態統一為 float，避免 object 型態造成計算錯誤。

    Args:
        df: 含有標準欄位（或可重新命名為標準欄位）的 DataFrame。

    Returns:
        標準 OHLCV DataFrame，index 為 DatetimeIndex，依日期排序。
    """
    df = df.copy()

    # index 統一為 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 數值型態統一為 float
    for col in _OHLCV_COLUMNS:
        df[col] = df[col].astype(float)

    df = df[_OHLCV_COLUMNS].sort_index()
    df.index.name = "date"
    return df


def get_ohlcv(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """透過 FinMind 抓取台股日線資料並轉為標準 OHLCV DataFrame。

    Args:
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        標準 OHLCV DataFrame。

    Raises:
        ValueError: 當 FinMind 回傳空資料時。
    """
    loader = DataLoader()

    # 若有 token 則登入，可提升 API 流量上限
    token = os.getenv("FINMIND_TOKEN")
    if token:
        loader.login_by_token(api_token=token)

    raw = loader.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"FinMind 回傳空資料：stock_id={stock_id}, "
            f"{start_date} ~ {end_date}"
        )

    raw = raw.rename(columns=_COLUMN_MAP)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")

    return _to_standard_ohlcv(raw)


def load_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """載入台股日線資料，優先讀取本地快取 CSV，否則從 FinMind 抓取。

    CSV 路徑為 data/{stock_id}.csv。
    若快取完整涵蓋指定期間則直接讀取；若範圍不足，會從 FinMind
    補抓指定期間並合併快取。回傳資料一律裁切至指定日期範圍。

    Args:
        stock_id: 股票代號，例如 "2330"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        標準 OHLCV DataFrame，index 為 DatetimeIndex，
        欄位為 open/high/low/close/volume（皆為 float）。
    """
    csv_path = os.path.join("data", f"{stock_id}.csv")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    business_days = pd.bdate_range(start, end)
    expected_start = business_days.min() if not business_days.empty else start
    expected_end = business_days.max() if not business_days.empty else end

    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        cached = _to_standard_ohlcv(cached)

        cache_incomplete = (
            cached.empty
            or expected_start < cached.index.min()
            or expected_end > cached.index.max()
        )
        if not cache_incomplete:
            return cached.loc[start:end]

        fresh = get_ohlcv(stock_id, start_date, end_date)
        df = pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")]
        df = _to_standard_ohlcv(df)
        df.to_csv(csv_path)
        return df.loc[start:end]

    # 檔案不存在：從 FinMind 抓取
    df = get_ohlcv(stock_id, start_date, end_date)

    os.makedirs("data", exist_ok=True)
    df.to_csv(csv_path)

    return df.loc[start:end]


def load_multiple(stock_ids: list, start_date: str,
                  end_date: str) -> dict:
    """批次載入多檔股票，回傳以股票代號為 key 的 DataFrame 字典。

    逐檔呼叫 load_data，沿用其本地 CSV 快取與 FinMind 抓取邏輯。
    單檔載入失敗時印出警告並略過，不中斷其餘股票。

    Args:
        stock_ids: 股票代號清單，例如 ["2330", "2317"]。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        dict[str, pd.DataFrame]，key 為股票代號，
        value 為標準 OHLCV DataFrame。
    """
    data = {}
    for stock_id in stock_ids:
        try:
            data[stock_id] = load_data(stock_id, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 載入 {stock_id} 失敗，已略過：{exc}")
    return data


# --- 處置股名單 / 上市櫃對照表 -------------------------------------------

# 處置股名單快取路徑；_META 記錄「已抓取的日期範圍」，
# 因為處置事件是稀疏的，無法用資料本身的最早/最晚日期推斷涵蓋範圍。
_DISPOSITION_CSV = os.path.join("data", "disposition_events.csv")
_DISPOSITION_META = os.path.join("data", "disposition_events_range.json")
_STOCK_INFO_CSV = os.path.join("data", "stock_info.csv")
_TRADING_DATE_CSV = os.path.join("data", "trading_dates.csv")
_TRADING_DATE_META = os.path.join("data", "trading_dates_range.json")

# 分批抓取的區間長度與批次間延遲（秒），避免撞到 600 次/hr 流量上限
_CHUNK_YEARS = 1
_REQUEST_DELAY = 1.0


def _get_loader() -> DataLoader:
    """建立 FinMind DataLoader，若 .env 有 token 則登入。

    Returns:
        已（視情況）登入的 DataLoader。
    """
    loader = DataLoader()
    token = os.getenv("FINMIND_TOKEN")
    if token:
        loader.login_by_token(api_token=token)
    return loader


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """將日期區間切成數個不重疊的子區間，用於分批打 API。

    Args:
        start: 起始日期。
        end: 結束日期。

    Returns:
        list[tuple[str, str]]，每個元素為 (start_date, end_date) 字串。
    """
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(
            cursor + pd.DateOffset(years=_CHUNK_YEARS) - pd.Timedelta(days=1),
            end,
        )
        chunks.append((cursor.strftime("%Y-%m-%d"),
                       chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_disposition_events(start_date: str,
                             end_date: str) -> pd.DataFrame:
    """抓取 FinMind 公布處置有價證券表（處置股名單）。

    使用 FinMind 套件內建的 taiwan_stock_disposition_securities_period
    方法（不帶 stock_id 代表抓取全市場）。此為 Sponsor 付費限定資料集，
    需在 .env 設定 FINMIND_TOKEN。

    因區間可能長達數年，會以每 _CHUNK_YEARS 年為單位分批抓取，
    批次之間加入短暫延遲。結果快取於 data/disposition_events.csv，
    並以 sidecar JSON 記錄已抓取的日期範圍；若快取涵蓋指定期間，
    則直接讀取不重打 API。

    Args:
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        處置事件 DataFrame，已裁切至指定日期範圍，依 date/stock_id 排序。
        欄位以 FinMind 實際回傳為準，預期包含 date, stock_id, stock_name,
        disposition_cnt, condition, measure, period_start, period_end。

    Raises:
        ValueError: 當所有批次皆回傳空資料時。
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # 快取涵蓋指定期間則直接讀取
    if os.path.exists(_DISPOSITION_CSV) and os.path.exists(_DISPOSITION_META):
        with open(_DISPOSITION_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        cached_start = pd.Timestamp(meta["start_date"])
        cached_end = pd.Timestamp(meta["end_date"])

        if cached_start <= start and cached_end >= end:
            cached = pd.read_csv(_DISPOSITION_CSV, dtype={"stock_id": str})
            cached["date"] = pd.to_datetime(cached["date"])
            mask = (cached["date"] >= start) & (cached["date"] <= end)
            print(f"[快取] 讀取 {_DISPOSITION_CSV}"
                  f"（涵蓋 {meta['start_date']} ~ {meta['end_date']}）")
            return cached.loc[mask].reset_index(drop=True)

    loader = _get_loader()
    chunks = _date_chunks(start, end)
    frames = []

    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"[抓取] 處置股名單 {chunk_start} ~ {chunk_end} "
              f"（{idx}/{len(chunks)}）")
        raw = loader.taiwan_stock_disposition_securities_period(
            start_date=chunk_start,
            end_date=chunk_end,
        )
        if raw is not None and not raw.empty:
            frames.append(raw)

        if idx < len(chunks):
            time.sleep(_REQUEST_DELAY)

    if not frames:
        raise ValueError(
            f"FinMind 回傳空資料：處置股名單 {start_date} ~ {end_date}"
        )

    df = pd.concat(frames, ignore_index=True)
    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(_DISPOSITION_CSV, index=False)
    with open(_DISPOSITION_META, "w", encoding="utf-8") as fh:
        json.dump({"start_date": start_date, "end_date": end_date}, fh)

    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


# 用來驗證交易日曆的參照股票（大型權值股，正常情況下每個交易日都有量）。
# 若這些股票在某日全數無價，該日必為休市，而非個別停牌。
_CALENDAR_REFERENCE_STOCKS = ("2330", "2317", "2454", "2412", "1301")


def _find_phantom_trading_days(dates: pd.DatetimeIndex, start_date: str,
                               end_date: str) -> pd.DatetimeIndex:
    """找出交易日曆列為交易日、但實際上全市場沒有交易的日期。

    FinMind 的 TaiwanStockTradingDate 已知會誤列休市日（例如
    2026-07-10）。以多檔大型權值股的實際價格資料交叉驗證：
    只有當所有參照股票在該日皆無資料時，才判定為日曆錯誤。

    Args:
        dates: 待驗證的交易日曆。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        應從日曆剔除的日期。
    """
    traded = None
    for stock_id in _CALENDAR_REFERENCE_STOCKS:
        try:
            index = load_data(stock_id, start_date, end_date).index
        except Exception:  # noqa: BLE001
            continue
        traded = index if traded is None else traded.union(index)

    if traded is None or traded.empty:
        print("[警告] 參照股票皆無法載入，跳過交易日曆驗證")
        return pd.DatetimeIndex([])

    # 參照股票資料涵蓋範圍之外的日期無從驗證，不予判定
    in_range = dates[(dates >= traded.min()) & (dates <= traded.max())]
    return in_range.difference(traded)


def fetch_trading_calendar(start_date: str, end_date: str,
                           validate: bool = True) -> pd.DatetimeIndex:
    """抓取 FinMind TaiwanStockTradingDate，取得真實台股交易日曆。

    使用專用的交易日資料集，已排除週末與國定假日，
    不需借用個股日K的日期 index 反推。結果快取於 data/trading_dates.csv。

    Args:
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。

    Returns:
        DatetimeIndex，已排序的交易日，裁切至指定範圍。

    Raises:
        ValueError: 當 FinMind 回傳空資料時。
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if os.path.exists(_TRADING_DATE_CSV) and os.path.exists(_TRADING_DATE_META):
        with open(_TRADING_DATE_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        cached_start = pd.Timestamp(meta["start_date"])
        cached_end = pd.Timestamp(meta["end_date"])
        today = pd.Timestamp.today().normalize()

        # 快取範圍若延伸到今天（含）之後，尾端必然還沒發生完，
        # 不能當成完整資料重複使用，否則日曆會永遠停在首次抓取那天。
        cache_usable = (cached_start <= start and cached_end >= end
                        and cached_end < today)
        if cache_usable:
            cached = pd.read_csv(_TRADING_DATE_CSV)
            dates = pd.DatetimeIndex(pd.to_datetime(cached["date"])).sort_values()
            print(f"[快取] 讀取 {_TRADING_DATE_CSV}"
                  f"（涵蓋 {meta['start_date']} ~ {meta['end_date']}）")
            dates = dates[(dates >= start) & (dates <= end)]
            return _validated(dates, start_date, end_date, validate)

    print(f"[抓取] 交易日曆 {start_date} ~ {end_date}")
    loader = _get_loader()
    df = loader.taiwan_stock_trading_date(
        start_date=start_date,
        end_date=end_date,
    )

    if df is None or df.empty:
        raise ValueError(
            f"FinMind 回傳空資料：交易日曆 {start_date} ~ {end_date}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date")

    os.makedirs("data", exist_ok=True)
    df[["date"]].to_csv(_TRADING_DATE_CSV, index=False)
    with open(_TRADING_DATE_META, "w", encoding="utf-8") as fh:
        json.dump({"start_date": start_date, "end_date": end_date}, fh)

    dates = pd.DatetimeIndex(df["date"])
    dates = dates[(dates >= start) & (dates <= end)]
    return _validated(dates, start_date, end_date, validate)


def _validated(dates: pd.DatetimeIndex, start_date: str, end_date: str,
               validate: bool) -> pd.DatetimeIndex:
    """以實際價格資料驗證交易日曆，剔除誤列的休市日。

    Args:
        dates: 待驗證的交易日曆。
        start_date: 起始日期。
        end_date: 結束日期。
        validate: 為 False 時直接回傳原日曆，不做驗證。

    Returns:
        剔除幽靈交易日後的日曆。
    """
    if not validate or dates.empty:
        return dates

    phantom = _find_phantom_trading_days(dates, start_date, end_date)
    if len(phantom):
        listed = "、".join(d.strftime("%Y-%m-%d") for d in phantom)
        print(f"[修正] 交易日曆剔除 {len(phantom)} 個幽靈交易日"
              f"（全市場無交易資料）：{listed}")
        dates = dates.difference(phantom)
    return dates


def fetch_stock_market_type() -> pd.DataFrame:
    """抓取 FinMind TaiwanStockInfo，取得 stock_id -> type 對照表。

    TaiwanStockInfo 為免費資料集。結果快取於 data/stock_info.csv，
    檔案存在時直接讀取。

    Returns:
        個股基本資料 DataFrame，欄位以 FinMind 實際回傳為準，
        預期包含 stock_id, stock_name, type（上市/上櫃別）等。

    Raises:
        ValueError: 當 FinMind 回傳空資料時。
    """
    if os.path.exists(_STOCK_INFO_CSV):
        print(f"[快取] 讀取 {_STOCK_INFO_CSV}")
        return pd.read_csv(_STOCK_INFO_CSV, dtype={"stock_id": str})

    print("[抓取] TaiwanStockInfo")
    loader = _get_loader()
    df = loader.taiwan_stock_info()

    if df is None or df.empty:
        raise ValueError("FinMind 回傳空資料：TaiwanStockInfo")

    df["stock_id"] = df["stock_id"].astype(str)

    os.makedirs("data", exist_ok=True)
    df.to_csv(_STOCK_INFO_CSV, index=False)

    return df


# --- 注意股名單（TWSE / TPEx 官網歷史端點）-------------------------------

# 官網帶日期歷史端點（非 OpenAPI；OpenAPI 只有當天）。
# TWSE 用 startDate/endDate + YYYYMMDD、扁平 JSON；
# TPEx 用 startDate/endDate + YYYY/MM/DD、巢狀 {tables:[{data}]}。
_ATTENTION_TWSE_URL = "https://www.twse.com.tw/rwd/zh/announcement/notice"
_ATTENTION_TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/bulletin/attention"
_ATTENTION_HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}

# 抓取節流：實測 TWSE 有硬限流（0.3s 下約 50 次即被 307 擋且持續數分鐘），
# 1.2s 下 300+ 次無虞；TPEx 0.3s 下 366 次零異常。兩所各自設定。
_ATTENTION_TWSE_DELAY = 1.2
_ATTENTION_TPEX_DELAY = 0.3
_ATTENTION_TIMEOUT = 25

# 被限流時的退避重試（TWSE 用；非 200 視為限流，不是「當天無資料」）
_ATTENTION_MAX_RETRY = 3
_ATTENTION_BACKOFF = (30, 60, 120)

# 款號 regex：抽「第N款」。實測兩所的括號寫法不一致——TWSE 多用全形﹝﹞、
# TPEx 混用全形（與半形)（例如當沖第十三款寫成「（第十三款)」），故括號
# 一律容忍全形/半形兩種形式，否則會漏抽（2020 未暴露，全歷史才顯現）。
# 一筆注意資訊常含多個款號，用 findall 全抽。款號「意義」仍依 market 區分。
_CN_DIGITS = "一二三四五六七八九十百零"
_BRACKET_OPEN = "﹝（("
_BRACKET_CLOSE = "﹞）)"
_RE_TWSE_CLAUSE = re.compile(
    f"[{_BRACKET_OPEN}]第([{_CN_DIGITS}]+)款[{_BRACKET_CLOSE}]")
_RE_TPEX_CLAUSE = re.compile(
    f"[{_BRACKET_OPEN}]第([{_CN_DIGITS}]+)款[{_BRACKET_CLOSE}]")

# multi-hot 欄位上限（TWSE 觀察到最高第十三款、TPEx 第十二款，取 14 涵蓋）
_MAX_ATTENTION_CLAUSE = 14


def _attention_cols(market: str) -> list:
    """回傳帶市場前綴的 multi-hot 欄名清單。

    TWSE 與 TPEx 的第 N 款定義不同，欄名帶前綴（twse_attention_N /
    tpex_attention_N）以避免兩所資料 concat 時混淆同編號的款號。

    Args:
        market: "twse" 或 "tpex"。

    Returns:
        [f"{market}_attention_1", …, f"{market}_attention_14"]。
    """
    return [f"{market}_attention_{i}"
            for i in range(1, _MAX_ATTENTION_CLAUSE + 1)]


def _cn_numeral_to_int(s: str) -> int:
    """將中文數字（一~十四）轉為整數。

    僅支援注意款號會用到的 1~99 範圍（實務只到十幾款）。

    Args:
        s: 中文數字字串，如 "一"、"十"、"十三"。

    Returns:
        對應整數；無法解析時回傳 0。
    """
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "零": 0}
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(s, 0)


def _parse_clauses(info: str, clause_re: re.Pattern) -> set:
    """從注意交易資訊文字抽出全部款號（整數集合）。

    Args:
        info: 「注意交易資訊」欄位文字。
        clause_re: 對應市場的款號 regex（全形或半形）。

    Returns:
        該筆命中的款號整數集合；抽不到則為空集合。
    """
    return {_cn_numeral_to_int(m) for m in clause_re.findall(info or "")}


def _clean_numeric(series: pd.Series) -> pd.Series:
    """將含哨兵值（如「-----」、空值）的字串欄轉為 float。

    可轉數字者轉 float，哨兵/空值一律轉 NaN（不以 0 頂替）。

    Args:
        series: 原始字串 Series。

    Returns:
        float64 Series。
    """
    cleaned = (series.astype(str)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def _rows_to_attention_df(records: list, market: str) -> pd.DataFrame:
    """將逐日抓下的原始列組成 multi-hot DataFrame（乾淨格式）。

    一列代表一個 (market, date, stock_id) 組合；同一組合若重複出現，
    合併其款號（布林做 OR）。款號欄名直接帶市場前綴
    ({market}_attention_N)；close_price/pe 直接轉 float、哨兵值為 NaN。

    Args:
        records: 每筆為 dict，含 date/stock_id/stock_name/cumulative_count/
            close_price/pe/raw_info/clauses(set)。
        market: "twse" 或 "tpex"。

    Returns:
        欄位為 market, date, stock_id, stock_name, cumulative_count,
        close_price(float), pe(float), {market}_attention_1..14(bool),
        raw_info 的 DataFrame。
    """
    att_cols = _attention_cols(market)

    merged = {}
    for r in records:
        key = (r["date"], r["stock_id"])
        if key not in merged:
            merged[key] = r
            merged[key]["clauses"] = set(r["clauses"])
        else:
            merged[key]["clauses"] |= r["clauses"]

    rows = []
    for (date, stock_id), r in merged.items():
        flags = {col: False for col in att_cols}
        for c in r["clauses"]:
            if 1 <= c <= _MAX_ATTENTION_CLAUSE:
                flags[f"{market}_attention_{c}"] = True
        rows.append({
            "market": market,
            "date": date,
            "stock_id": stock_id,
            "stock_name": r["stock_name"],
            "cumulative_count": r["cumulative_count"],
            "close_price": r["close_price"],
            "pe": r["pe"],
            **flags,
            "raw_info": r["raw_info"],
        })

    cols = (["market", "date", "stock_id", "stock_name", "cumulative_count",
             "close_price", "pe"] + att_cols + ["raw_info"])
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)
        # close_price / pe 直接清洗成 float，哨兵值 -> NaN
        df["close_price"] = _clean_numeric(df["close_price"])
        df["pe"] = _clean_numeric(df["pe"])
    return df


def _attention_http_get(url: str, params: dict, label: str,
                        allow_redirects: bool) -> dict:
    """對注意股端點做 GET，含退避重試（涵蓋限流與網路例外）。

    兩類失敗都退避重試：非 200 回應（TWSE 限流回 307）與網路例外
    （逾時、連線中斷）。單一暫時性 timeout 不應中止整批抓取。

    Args:
        url: 端點網址。
        params: 查詢參數。
        label: 記錄用標籤（如 "TWSE 20200316"）。
        allow_redirects: 是否跟隨轉址（TWSE 設 False 以偵測 307 限流）。

    Returns:
        回應的 JSON dict。

    Raises:
        RuntimeError: 重試用盡仍失敗。
    """
    reason = "?"
    for attempt in range(_ATTENTION_MAX_RETRY + 1):
        try:
            resp = requests.get(url, params=params, headers=_ATTENTION_HEADERS,
                                timeout=_ATTENTION_TIMEOUT,
                                allow_redirects=allow_redirects)
            if resp.status_code == 200:
                return resp.json()
            reason = f"HTTP {resp.status_code}"          # 非 200：多為限流
        except requests.exceptions.RequestException as exc:
            reason = type(exc).__name__                   # 逾時/連線中斷
        if attempt < _ATTENTION_MAX_RETRY:
            wait = _ATTENTION_BACKOFF[attempt]
            print(f"[重試] {label} {reason}，{wait}s 後重試"
                  f"（第 {attempt + 1} 次）")
            time.sleep(wait)
    raise RuntimeError(f"{label} 重試 {_ATTENTION_MAX_RETRY} 次仍失敗（{reason}）")


def _fetch_twse_attention_day(date: pd.Timestamp) -> list:
    """抓 TWSE 單日注意股，含限流／網路例外退避重試。

    非 200 視為限流（不是當天無資料），與逾時／連線中斷一律退避重試。

    Args:
        date: 目標日期。

    Returns:
        該日原始列的 list（每筆 dict）。
    """
    ymd = date.strftime("%Y%m%d")
    params = {"startDate": ymd, "endDate": ymd, "response": "json"}
    j = _attention_http_get(_ATTENTION_TWSE_URL, params, f"TWSE {ymd}",
                            allow_redirects=False)
    return _twse_rows(j.get("data", []), date)


def _twse_rows(data: list, date: pd.Timestamp) -> list:
    """將 TWSE 原始 data 列轉為標準 record（欄位順序見探測報告）。"""
    out = []
    iso = date.strftime("%Y-%m-%d")
    for row in data:
        # 欄位: 編號,證券代號,證券名稱,累計次數,注意交易資訊,日期,收盤價,本益比
        if len(row) < 8:
            continue
        out.append({
            "date": iso,
            "stock_id": str(row[1]),
            "stock_name": row[2],
            "cumulative_count": row[3],
            "close_price": row[6],
            "pe": row[7],
            "raw_info": row[4],
            "clauses": _parse_clauses(str(row[4]), _RE_TWSE_CLAUSE),
        })
    return out


def _fetch_tpex_attention_day(date: pd.Timestamp) -> list:
    """抓 TPEx 單日注意股（無限流問題，但仍對網路例外退避重試）。

    Args:
        date: 目標日期。

    Returns:
        該日原始列的 list（每筆 dict）。
    """
    slash = date.strftime("%Y/%m/%d")
    params = {"startDate": slash, "endDate": slash, "response": "json"}
    j = _attention_http_get(_ATTENTION_TPEX_URL, params, f"TPEx {slash}",
                            allow_redirects=True)
    tables = j.get("tables", [])
    data = tables[0].get("data", []) if tables else []
    out = []
    iso = date.strftime("%Y-%m-%d")
    for row in data:
        # 欄位: 編號,證券代號,證券名稱,累計,注意交易資訊,公告日期,收盤價,本益比,link
        if len(row) < 8:
            continue
        out.append({
            "date": iso,
            "stock_id": str(row[1]),
            "stock_name": row[2],
            "cumulative_count": row[3],
            "close_price": row[6],
            "pe": row[7],
            "raw_info": row[4],
            "clauses": _parse_clauses(str(row[4]), _RE_TPEX_CLAUSE),
        })
    return out


def _fetch_attention_year(year: int, market: str, csv_path: str,
                          day_fetcher, delay: float,
                          cap_end: str = None) -> pd.DataFrame:
    """逐交易日抓取指定年份注意股，組成 multi-hot DataFrame 並快取。

    Args:
        year: 西元年。
        market: "twse" 或 "tpex"。
        csv_path: 快取路徑。
        day_fetcher: 單日抓取函式（回傳 record list）。
        delay: 每次請求後延遲秒數。
        cap_end: 選填，當年只抓到此日期為止（含），格式 "YYYY-MM-DD"。
            用於最後一年對齊資料範圍（如 2026 只到 2026-07-15）。

    Returns:
        multi-hot DataFrame。
    """
    if os.path.exists(csv_path):
        print(f"[快取] 讀取 {csv_path}")
        return pd.read_csv(csv_path, dtype={"stock_id": str})

    calendar = fetch_trading_calendar(f"{year}-01-01", f"{year}-12-31")
    if cap_end:
        calendar = calendar[calendar <= pd.Timestamp(cap_end)]
    print(f"[抓取] {market.upper()} 注意股 {year}（{len(calendar)} 交易日，"
          f"delay={delay}s{f'，封頂 {cap_end}' if cap_end else ''}）")

    records = []
    for i, day in enumerate(calendar, 1):
        records.extend(day_fetcher(day))
        if i % 50 == 0 or i == len(calendar):
            print(f"  [{i}/{len(calendar)}] {day.date()} 累計 {len(records)} 筆")
        time.sleep(delay)

    df = _rows_to_attention_df(records, market)
    os.makedirs("data", exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def fetch_attention_events_twse(year: int,
                                cap_end: str = None) -> pd.DataFrame:
    """抓取指定年份 TWSE（上市）注意股名單，multi-hot 款號編碼。

    一列代表一個 (market, date, stock_id) 組合，attention_1~14 為布林，
    以 TWSE 全形款號 regex 抽出的全部款號標 True。TWSE 官網有硬限流，
    採 delay=1.2s 並對 307 退避重試。快取於
    data/attention_events_twse_{year}.csv。

    注意：TWSE 與 TPEx 的第 N 款定義不同，market 欄位是關鍵區分依據。

    Args:
        year: 西元年。
        cap_end: 選填，當年只抓到此日期為止（含），格式 "YYYY-MM-DD"。

    Returns:
        multi-hot DataFrame。
    """
    path = os.path.join("data", f"attention_events_twse_{year}.csv")
    return _fetch_attention_year(year, "twse", path,
                                 _fetch_twse_attention_day,
                                 _ATTENTION_TWSE_DELAY, cap_end)


def fetch_attention_events_tpex(year: int,
                                cap_end: str = None) -> pd.DataFrame:
    """抓取指定年份 TPEx（上櫃）注意股名單，multi-hot 款號編碼。

    一列代表一個 (market, date, stock_id) 組合，attention_1~14 為布林，
    以 TPEx 半形款號 regex 抽出的全部款號標 True。TPEx 無限流問題，
    採 delay=0.3s。快取於 data/attention_events_tpex_{year}.csv。

    注意：TWSE 與 TPEx 的第 N 款定義不同，market 欄位是關鍵區分依據。

    Args:
        year: 西元年。
        cap_end: 選填，當年只抓到此日期為止（含），格式 "YYYY-MM-DD"。

    Returns:
        multi-hot DataFrame。
    """
    path = os.path.join("data", f"attention_events_tpex_{year}.csv")
    return _fetch_attention_year(year, "tpex", path,
                                 _fetch_tpex_attention_day,
                                 _ATTENTION_TPEX_DELAY, cap_end)


def fetch_attention_events_range(start_date: str, end_date: str,
                                 markets=("twse", "tpex")) -> dict:
    """抓取日期範圍內的注意股，分年、分市場快取（比照處置股分年風格）。

    逐年呼叫對應的單年函式，已存在的年份檔案會直接讀取跳過，方便中斷
    後續抓。最後一年以 end_date 封頂（對齊處置股資料範圍）。TWSE 與
    TPEx 各自使用實測安全的節流值，互不共用。

    Args:
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 結束日期，格式 "YYYY-MM-DD"。
        markets: 要抓的市場，預設兩所都抓。

    Returns:
        dict[str, pd.DataFrame]，key 為 "twse"/"tpex"，value 為該所全期間
        合併後的 DataFrame。
    """
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    fetchers = {"twse": fetch_attention_events_twse,
                "tpex": fetch_attention_events_tpex}

    out = {}
    for market in markets:
        fetcher = fetchers[market]
        frames = []
        for year in range(start_year, end_year + 1):
            cap = end_date if year == end_year else None
            frames.append(fetcher(year, cap_end=cap))
        out[market] = pd.concat(frames, ignore_index=True)
    return out
