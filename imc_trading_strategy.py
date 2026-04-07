"""
IMC Prosperity Algo Trading Strategy
=====================================
Products  : EMERALDS, TOMATOES
Strategy  : Market-Making + Order Book Imbalance Signal
Author    : Sample Work
Data      : prices_round_0_day_*.csv  /  trades_round_0_day_*.csv

HOW TO RUN
----------
    pip install pandas numpy matplotlib
    python imc_trading_strategy.py

The script will:
  1. Load and clean all price + trade CSVs
  2. Run the market-making strategy on each product
  3. Print a P&L summary
  4. Save a performance chart as  results.png
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────

def load_data(price_pattern="prices_round_*.csv",
              trade_pattern="trades_round_*.csv"):
    """
    Reads all price and trade CSV files that match the given glob patterns.
    Returns two DataFrames sorted by day + timestamp.
    """
    # --- prices ---
    price_files = glob.glob(price_pattern)
    if not price_files:
        # fallback: look in uploads folder
        price_files = glob.glob(f"/mnt/user-data/uploads/{price_pattern}")

    prices = pd.concat(
        [pd.read_csv(f, sep=";", dtype_backend="numpy_nullable") for f in price_files],
        ignore_index=True
    ).sort_values(["day", "timestamp"]).reset_index(drop=True)

    # --- trades ---
    trade_files = glob.glob(trade_pattern)
    if not trade_files:
        trade_files = glob.glob(f"/mnt/user-data/uploads/{trade_pattern}")

    trades = pd.concat(
        [pd.read_csv(f, sep=";", dtype_backend="numpy_nullable") for f in trade_files],
        ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(prices):,} price rows from {len(price_files)} file(s)")
    print(f"Loaded {len(trades):,} trade rows from {len(trade_files)} file(s)")
    print(f"Products : {sorted(prices['product'].unique())}")
    print(f"Days     : {sorted(prices['day'].unique())}\n")

    return prices, trades


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds useful columns to the price DataFrame:
      - spread          : best ask minus best bid
      - mid_price       : already in data, but recalculated as a sanity check
      - imbalance       : (total bid vol - total ask vol) / total vol
                          Positive  → more buyers  → price likely to rise
                          Negative  → more sellers → price likely to fall
      - rolling_mid     : 20-period rolling average of mid price (trend filter)
    """
    df = df.copy()

    # Spread between best bid and best ask
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    # Total volume on each side (up to 3 levels)
    df["bid_vol"] = (df["bid_volume_1"].fillna(0)
                     + df["bid_volume_2"].fillna(0)
                     + df["bid_volume_3"].fillna(0))
    df["ask_vol"] = (df["ask_volume_1"].fillna(0)
                     + df["ask_volume_2"].fillna(0)
                     + df["ask_volume_3"].fillna(0))

    total_vol = df["bid_vol"] + df["ask_vol"]

    # Order book imbalance  [-1 … +1]
    df["imbalance"] = np.where(
        total_vol > 0,
        (df["bid_vol"] - df["ask_vol"]) / total_vol,
        0
    )

    # 20-step rolling mid price per product (trend reference)
    df["rolling_mid"] = (
        df.groupby("product")["mid_price"]
          .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )

    return df


# ─────────────────────────────────────────────
# 3.  MARKET-MAKING STRATEGY
# ─────────────────────────────────────────────

class MarketMaker:
    """
    Simple market-making bot.

    Logic
    -----
    At every timestep the bot decides whether to BUY, SELL, or HOLD:

    BUY  signal  →  imbalance > threshold  (more buyers than sellers)
                    AND mid_price <= rolling average  (not already too high)
    SELL signal  →  imbalance < -threshold (more sellers than buyers)
                    AND mid_price >= rolling average  (not already too low)

    Position limits keep the bot from taking on unlimited risk.
    P&L is tracked in SEASHELLS (the competition currency).
    """

    def __init__(self,
                 product: str,
                 position_limit: int = 20,
                 imbalance_threshold: float = 0.02,
                 trade_size: int = 3):
        self.product            = product
        self.position_limit     = position_limit
        self.imbalance_threshold = imbalance_threshold
        self.trade_size         = trade_size

        # state
        self.position   = 0       # current holding (+ = long, - = short)
        self.cash       = 0.0     # realised cash flow
        self.trades_log = []      # list of executed trades for analysis

    # ── decision logic ──────────────────────────────────────────────
    def decide(self, row: pd.Series) -> str:
        """
        Returns  'BUY', 'SELL', or 'HOLD'
        given the current order-book snapshot.
        """
        imb = row["imbalance"]
        mid = row["mid_price"]
        avg = row["rolling_mid"]

        if (imb > self.imbalance_threshold
                and mid <= avg
                and self.position < self.position_limit):
            return "BUY"

        if (imb < -self.imbalance_threshold
                and mid >= avg
                and self.position > -self.position_limit):
            return "SELL"

        return "HOLD"

    # ── execute a single step ────────────────────────────────────────
    def step(self, row: pd.Series):
        """
        Runs decide(), then simulates execution:
          BUY  → we hit the best ask price
          SELL → we hit the best bid price
        """
        action = self.decide(row)

        if action == "BUY":
            exec_price = row["ask_price_1"]          # we pay the ask
            qty        = min(self.trade_size,
                             self.position_limit - self.position)
            self.position += qty
            self.cash     -= exec_price * qty

        elif action == "SELL":
            exec_price = row["bid_price_1"]          # we receive the bid
            qty        = min(self.trade_size,
                             self.position_limit + self.position)
            self.position -= qty
            self.cash     += exec_price * qty

        else:
            exec_price = None
            qty        = 0

        # mark-to-market P&L = cash in hand + current value of open position
        mtm_pnl = self.cash + self.position * row["mid_price"]

        self.trades_log.append({
            "timestamp"  : row["timestamp"],
            "day"        : row["day"],
            "action"     : action,
            "price"      : exec_price,
            "qty"        : qty,
            "position"   : self.position,
            "cash"       : self.cash,
            "mid_price"  : row["mid_price"],
            "pnl"        : mtm_pnl,
        })

    # ── run over full price series ───────────────────────────────────
    def run(self, df: pd.DataFrame):
        """
        Iterates through every row of the (filtered + featured) price
        DataFrame and calls step() at each timestamp.
        """
        product_df = df[df["product"] == self.product].copy()
        for _, row in product_df.iterrows():
            self.step(row)
        return pd.DataFrame(self.trades_log)

    # ── summary stats ────────────────────────────────────────────────
    def summary(self) -> dict:
        log = pd.DataFrame(self.trades_log)
        trades_only = log[log["action"] != "HOLD"]
        final_pnl   = log["pnl"].iloc[-1] if len(log) else 0

        return {
            "product"       : self.product,
            "final_pnl"     : round(final_pnl, 2),
            "total_trades"  : len(trades_only),
            "buys"          : (trades_only["action"] == "BUY").sum(),
            "sells"         : (trades_only["action"] == "SELL").sum(),
            "final_position": self.position,
            "final_cash"    : round(self.cash, 2),
        }


# ─────────────────────────────────────────────
# 4.  RESULTS & CHART
# ─────────────────────────────────────────────

def plot_results(logs: dict, prices: pd.DataFrame, save_path="results.png"):
    """
    Plots, for each product:
      - Mid price over time
      - Buy / Sell markers
      - Cumulative P&L curve
    """
    products = list(logs.keys())
    n        = len(products)

    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]   # keep indexing consistent

    fig.suptitle("IMC Prosperity — Market-Making Strategy Results",
                 fontsize=15, fontweight="bold", y=1.01)

    for i, product in enumerate(products):
        log = logs[product]
        ax_price, ax_pnl = axes[i]

        # ── left: price + trade signals ──────────────────────────
        ax_price.plot(log["timestamp"], log["mid_price"],
                      color="#4C72B0", linewidth=1, label="Mid Price")

        buys  = log[log["action"] == "BUY"]
        sells = log[log["action"] == "SELL"]

        ax_price.scatter(buys["timestamp"], buys["mid_price"],
                         marker="^", color="green", s=60,
                         zorder=5, label="BUY")
        ax_price.scatter(sells["timestamp"], sells["mid_price"],
                         marker="v", color="red", s=60,
                         zorder=5, label="SELL")

        ax_price.set_title(f"{product} — Price & Signals", fontweight="bold")
        ax_price.set_xlabel("Timestamp")
        ax_price.set_ylabel("Price (SEASHELLS)")
        ax_price.legend()
        ax_price.grid(alpha=0.3)

        # ── right: cumulative P&L ─────────────────────────────────
        ax_pnl.plot(log["timestamp"], log["pnl"],
                    color="darkorange", linewidth=1.5, label="Mark-to-Market P&L")
        ax_pnl.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_pnl.fill_between(log["timestamp"], log["pnl"], 0,
                             where=log["pnl"] >= 0,
                             alpha=0.2, color="green", label="Profit")
        ax_pnl.fill_between(log["timestamp"], log["pnl"], 0,
                             where=log["pnl"] < 0,
                             alpha=0.2, color="red", label="Loss")

        ax_pnl.set_title(f"{product} — Cumulative P&L", fontweight="bold")
        ax_pnl.set_xlabel("Timestamp")
        ax_pnl.set_ylabel("P&L (SEASHELLS)")
        ax_pnl.legend()
        ax_pnl.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {save_path}")


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  IMC Prosperity — Market-Making Strategy")
    print("=" * 55 + "\n")

    # Load
    prices, trades = load_data()

    # Feature engineering
    prices = add_features(prices)

    # Run strategy for each product
    products = ["EMERALDS", "TOMATOES"]
    logs     = {}
    summaries = []

    for product in products:
        bot = MarketMaker(
            product             = product,
            position_limit      = 20,
            imbalance_threshold = 0.02,
            trade_size          = 3,
        )
        log          = bot.run(prices)
        logs[product] = log
        summaries.append(bot.summary())

    # Print summary table
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    summary_df = pd.DataFrame(summaries).set_index("product")
    print(summary_df.to_string())

    total_pnl = summary_df["final_pnl"].sum()
    print(f"\n  TOTAL P&L (both products): {total_pnl:,.2f} SEASHELLS")
    print("=" * 55)

    # Plot
    plot_results(logs, prices, save_path="results.png")

    print("\nDone! Check results.png for the performance chart.")


if __name__ == "__main__":
    main()
