"""資料載入模組。

負責從 FinMind 抓取台股日線資料，或從本地 CSV 讀取，
並統一輸出標準 OHLCV DataFrame。
另提供處置股名單與上市/上櫃對照表的抓取與快取。
"""

import json
import os
import time

import pandas as pd
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
