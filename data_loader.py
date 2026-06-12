"""資料載入模組。

負責從 FinMind 抓取台股日線資料，或從本地 CSV 讀取，
並統一輸出標準 OHLCV DataFrame。
"""

import os

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
