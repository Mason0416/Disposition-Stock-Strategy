"""處置股做多策略——研究圖表產出。

讀取 data/trade_level.csv 與重跑滑價敏感度矩陣，輸出四張 PNG 至 charts/。
純繪圖腳本，不含任何策略或回測邏輯（回測邏輯一律 import 自 event_backtest）。

用法：
    python make_charts.py
"""

import os

import matplotlib
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_loader import fetch_trading_calendar  # noqa: E402
from event_backtest import (  # noqa: E402
    CLEAN_CSV,
    SLIPPAGE_SCENARIOS,
    STOP_FILL_MODES,
    TRADE_CSV,
    apply_slippage,
    build_price_ranges,
    build_trade_level,
    load_prices,
    scenario_stats,
)

CHART_DIR = "charts"

# 調色盤（validated 參考調色盤；light surface）
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"   # 藍
SERIES_2 = "#008300"   # 綠
MUTED = "#a3a29c"      # 去強調用灰
GRID = "#e3e2dd"

# 統計上無顯著差異、且表現最佳的進場日區間
BEST_BAND = (2, 6)

plt.rcParams.update({
    "font.family": "PingFang HK",
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 130,
})


def _style(ax) -> None:
    """套用共用的座標軸樣式（隱藏上右邊框、格線僅留水平）。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(False)


def chart_entry_return_by_day(trades: pd.DataFrame) -> None:
    """長條圖：各進場日的平均報酬率，標示無顯著差異的最佳區間。"""
    stats = trades.groupby("entry_day_index")["return_pct"].agg(
        ["size", "mean", "std"]
    )
    mean = stats["mean"] * 100
    ci95 = 1.96 * stats["std"] / stats["size"] ** 0.5 * 100

    lo, hi = BEST_BAND
    colors = [SERIES_1 if lo <= i <= hi else MUTED for i in stats.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stats.index, mean, color=colors, width=0.68,
                  yerr=ci95, capsize=3,
                  error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2})
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)

    for day, value, err in zip(stats.index, mean, ci95):
        if value >= 0:
            y, va = value + err + 0.08, "bottom"
        else:
            y, va = value - err - 0.08, "top"
        ax.text(day, y, f"{value:.2f}", ha="center", va=va,
                fontsize=9, color=TEXT_SECONDARY)

    ax.set_xlabel("進場日（處置期間第幾個交易日）")
    ax.set_ylabel("平均報酬率 %")
    ax.set_title("進場時機對平均報酬率的影響", fontsize=14, pad=30,
                 color=TEXT_PRIMARY, loc="left")
    ax.text(0, 1.02,
            f"誤差線為 95% 信賴區間｜藍色為第 {lo}~{hi} 天（彼此無顯著差異）",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_xticks(list(stats.index))
    _style(ax)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "entry_return_by_day.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [輸出] {path}")


def chart_entry_winrate_stoprate(trades: pd.DataFrame) -> None:
    """折線圖：勝率與停損觸發率隨進場日的變化（共用 0~100% 軸）。"""
    grouped = trades.groupby("entry_day_index")
    win = grouped["return_pct"].apply(lambda s: (s > 0).mean()) * 100
    stop = grouped["exit_reason"].apply(lambda s: (s == "stop_loss").mean()) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(win.index, win, color=SERIES_1, linewidth=2, marker="o",
            markersize=6, label="勝率 %")
    ax.plot(stop.index, stop, color=SERIES_2, linewidth=2, marker="s",
            markersize=6, label="停損觸發率 %")

    ax.annotate("勝率", (win.index[-1], win.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=SERIES_1, fontsize=10,
                va="center")
    ax.annotate("停損觸發率", (stop.index[-1], stop.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=SERIES_2, fontsize=10,
                va="center")

    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, 12.6)
    ax.set_xticks(list(win.index))
    ax.set_xlabel("進場日（處置期間第幾個交易日）")
    ax.set_ylabel("百分比 %")
    ax.set_title("勝率與停損觸發率隨進場日的變化", fontsize=14, pad=30,
                 color=TEXT_PRIMARY, loc="left")
    ax.text(0, 1.02, "越晚進場曝險天數越少，被停損掃出的機會越低",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    _style(ax)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "entry_winrate_stoprate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [輸出] {path}")


def chart_slippage_sensitivity(matrix: pd.DataFrame) -> None:
    """折線圖：整體平均報酬率對滑價假設的敏感度，標示損益兩平線。"""
    labels = {"stop_line": "停損線成交", "gap_adjusted": "跳空調整成交"}
    colors = {"stop_line": SERIES_1, "gap_adjusted": SERIES_2}
    markers = {"stop_line": "o", "gap_adjusted": "s"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [s * 100 for s in SLIPPAGE_SCENARIOS]

    for mode in STOP_FILL_MODES:
        sub = matrix[matrix["mode"] == mode].sort_values("slippage")
        ax.plot(x, sub["平均%"], color=colors[mode], linewidth=2,
                marker=markers[mode], markersize=8, label=labels[mode])
        for xi, yi in zip(x, sub["平均%"]):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=colors[mode])

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1.4, linestyle="--")
    ax.annotate("損益兩平", (x[0], 0), xytext=(2, -14),
                textcoords="offset points", fontsize=9.5,
                color=TEXT_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}%" for v in x])
    ax.set_xlabel("滑價假設")
    ax.set_ylabel("整體平均報酬率 %")
    ax.set_title("滑價假設決定策略生死", fontsize=14, pad=30,
                 color=TEXT_PRIMARY, loc="left")
    ax.text(0, 1.02, "1% 滑價下平均報酬貼著損益兩平線，跳空調整後轉負",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=10)
    _style(ax)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "slippage_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [輸出] {path}")


def chart_return_distribution(trades: pd.DataFrame) -> None:
    """直方圖：報酬率分布，標示中位數與平均值。"""
    ret = trades["return_pct"] * 100
    lo, hi = -25, 40
    outside = ((ret < lo) | (ret > hi)).mean() * 100
    clipped = ret[(ret >= lo) & (ret <= hi)]

    median = ret.median()
    mean = ret.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clipped, bins=90, color=SERIES_1, alpha=0.85, edgecolor=SURFACE,
            linewidth=0.3)

    # 預留上方空間給標註；參考線只畫到 72% 高度，讓上方標註帶保持淨空
    peak_height = ax.get_ylim()[1]
    ax.set_ylim(0, peak_height * 1.3)
    top = ax.get_ylim()[1]

    ax.axvline(median, color=TEXT_PRIMARY, linewidth=2, ymax=0.72)
    ax.axvline(mean, color="#eb6834", linewidth=2, linestyle="--", ymax=0.72)

    ax.annotate(f"中位數 {median:.2f}%", (median, top * 0.60),
                xytext=(-104, 0), textcoords="offset points",
                fontsize=10, color=TEXT_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_PRIMARY,
                            "linewidth": 1})
    ax.annotate(f"平均 {mean:.2f}%", (mean, top * 0.45),
                xytext=(52, 0), textcoords="offset points",
                fontsize=10, color="#eb6834",
                arrowprops={"arrowstyle": "-", "color": "#eb6834",
                            "linewidth": 1})

    # 進場隔日即觸發停損者，出場價恰為 entry x 0.91，故在此群聚成尖峰
    stop_peak = clipped[(clipped > -10.5) & (clipped < -8.5)]
    if not stop_peak.empty:
        ax.annotate("停損出場群聚（隔日即觸發，固定約 -9.5%）",
                    (stop_peak.median(), peak_height * 1.02),
                    xytext=(30, 30), textcoords="offset points",
                    fontsize=9.5, color=TEXT_SECONDARY, ha="left",
                    arrowprops={"arrowstyle": "->", "color": TEXT_SECONDARY,
                                "linewidth": 1})

    ax.set_xlim(lo, hi)
    ax.set_xlabel("單筆報酬率 %")
    ax.set_ylabel("交易筆數")
    ax.set_title("報酬率分布右偏：平均值靠右尾少數大贏家撐起", fontsize=14,
                 pad=30, color=TEXT_PRIMARY, loc="left")
    ax.text(0, 1.02,
            f"滑價 0.1% 基準版本，{len(ret):,} 筆交易"
            f"（{outside:.2f}% 落在 {lo}%~{hi}% 之外未顯示）",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)
    _style(ax)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "return_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [輸出] {path}")


def build_sensitivity_matrix() -> pd.DataFrame:
    """重跑 3 種滑價 x 2 種停損成交模式的敏感度矩陣。

    Returns:
        含 mode / slippage / 績效指標的 DataFrame。
    """
    events = pd.read_csv(CLEAN_CSV, dtype={"stock_id": str})
    for col in ["period_start", "period_end"]:
        events[col] = pd.to_datetime(events[col])

    calendar = fetch_trading_calendar("2019-12-01", "2026-08-31")
    prices = load_prices(build_price_ranges(events, calendar))

    rows = []
    for mode in STOP_FILL_MODES:
        base = build_trade_level(events, prices, calendar,
                                 stop_fill_mode=mode, verbose=False)
        for slip in SLIPPAGE_SCENARIOS:
            stats = {"mode": mode, "slippage": slip}
            stats.update(scenario_stats(apply_slippage(base, slip)))
            rows.append(stats)
    return pd.DataFrame(rows)


def main() -> None:
    """產出全部研究圖表。"""
    load_dotenv()
    os.makedirs(CHART_DIR, exist_ok=True)

    trades = pd.read_csv(TRADE_CSV, dtype={"stock_id": str})
    print(f"讀取 {TRADE_CSV}（{len(trades)} 筆）")

    print("重跑滑價敏感度矩陣…")
    matrix = build_sensitivity_matrix()

    print("繪圖：")
    chart_entry_return_by_day(trades)
    chart_entry_winrate_stoprate(trades)
    chart_slippage_sensitivity(matrix)
    chart_return_distribution(trades)


if __name__ == "__main__":
    main()
