"""Single-stock moving-average crossover strategy."""

import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Generate 5/20-day moving-average crossover signals.

    A golden cross produces 1, a death cross produces -1, and all other
    dates produce 0. Signals are calculated after the T-1 close and executed
    at the T open, so the strategy does not use look-ahead information.

    Args:
        df: OHLCV DataFrame containing a ``close`` column.

    Returns:
        Integer Series with the same index as ``df`` and values in 1/-1/0.
    """
    close = df["close"]
    fast = close.rolling(window=5).mean()
    slow = close.rolling(window=20).mean()

    above = fast > slow
    prev_above = above.shift(1, fill_value=False)

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[(~prev_above) & above] = 1
    signals[prev_above & (~above)] = -1

    return signals.shift(1, fill_value=0).astype(int)
