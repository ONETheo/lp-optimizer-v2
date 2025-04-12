"""
Result Processor Module

Handles processing, visualization, and reporting of backtest results.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Dict, Any, Optional

# Use absolute imports
from src.utils.helpers import ensure_dir_exists, format_percentage, format_currency

logger = logging.getLogger(__name__)

class ResultProcessor:
    """Processes and visualizes backtest results."""

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the ResultProcessor.

        Args:
            output_dir (str): Directory to save plots and reports.
        """
        self.output_dir = output_dir
        ensure_dir_exists(self.output_dir)
        # Set a default plot style
        try:
            plt.style.use('seaborn-v0_8-darkgrid') # Use a modern seaborn style
        except OSError:
             plt.style.use('ggplot') # Fallback style
             logger.warning("Seaborn style 'seaborn-v0_8-darkgrid' not found, using 'ggplot'.")

        logger.info(f"ResultProcessor initialized. Output directory: {self.output_dir}")

    def print_summary_metrics(self, metrics: Dict[str, Any]):
        """Prints a formatted summary of key backtest metrics to the console."""
        print("\n--- Backtest Performance Summary ---")
        # Timing and Value
        print(f" Duration:                 {metrics.get('duration_days', 'N/A')} days")
        print(f" Initial Investment:       {format_currency(metrics.get('initial_investment', 0))}")
        print(f" Final Position Value:     {format_currency(metrics.get('final_position_value', 0))}")
        print(f" Total Fees Earned:        {format_currency(metrics.get('total_fees_earned', 0))}")
        print(f" Final Value (incl. Fees): {format_currency(metrics.get('final_value_incl_fees', 0))}")
        print("-" * 34)
        # Profitability
        print(f" Gross Profit:             {format_currency(metrics.get('gross_profit', 0))}")
        print(f" Total Transaction Costs:  {format_currency(metrics.get('total_tx_costs', 0))}")
        print(f"   - Gas Costs:            {format_currency(metrics.get('total_gas_costs', 0))}")
        print(f"   - Slippage Costs:       {format_currency(metrics.get('total_slippage_costs', 0))}")
        print(f" Net Profit (after costs): {format_currency(metrics.get('net_profit', 0))}")
        print("-" * 34)
        # APRs
        print(f" Gross APR:                {format_percentage(metrics.get('gross_apr', 0) / 100.0)}")
        print(f" Fees APR:                 {format_percentage(metrics.get('fees_apr', 0) / 100.0)}")
        print(f" Net APR (after costs):    {format_percentage(metrics.get('net_apr', 0) / 100.0)}")
        print("-" * 34)
        # Strategy Behavior
        print(f" Rebalance Count:          {metrics.get('rebalance_count', 'N/A')}")
        print(f" Rebalance Frequency:      {metrics.get('rebalance_frequency', 0):.2f} per day")
        print(f" Time in Position:         {format_percentage(metrics.get('time_in_position_pct', 0) / 100.0)}")
        print(f" Period Price Volatility:  {format_percentage(metrics.get('volatility_pct', 0) / 100.0)}") # Volatility for the period
        print("----------------------------------")


    def _generate_plot_title(self, parameters: Dict[str, Any]) -> str:
         """Helper to create a standardized plot title from parameters."""
         pool_id = parameters.get('pool_address', 'N/A')
         exchange = parameters.get('exchange', 'N/A').upper()
         t0 = parameters.get('token0', 'T0')
         t1 = parameters.get('token1', 'T1')
         width = parameters.get('initial_width_pct', 'N/A')
         rebal = parameters.get('rebalance_buffer_pct', 'N/A')
         cutoff = parameters.get('cutoff_buffer_pct', 'N/A')
         start = parameters.get('start_date', 'N/A')
         end = parameters.get('end_date', 'N/A')

         # Shorten pool address for title if needed
         short_pool_id = f"{pool_id[:6]}...{pool_id[-4:]}" if len(pool_id) > 15 else pool_id

         title = (f"{exchange} {t0}/{t1} ({short_pool_id})\n"
                  f"Strategy: Width={width}%, RebalBuf={rebal}%, CutoffBuf={cutoff}%\n"
                  f"Period: {start} to {end}")
         return title

    def plot_price_and_bounds(
        self,
        results_log: pd.DataFrame,
        parameters: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> None:
        """
        Plots the price history along with the LP position bounds and rebalance events.

        Args:
            results_log (pd.DataFrame): DataFrame containing the daily log from the backtest.
                                        Required columns: timestamp, price, lower_bound, upper_bound, rebalanced.
            parameters (Dict): Dictionary of backtest parameters for context in the title.
            output_file (Optional[str]): Path relative to output_dir to save the plot. If None, plot is shown.
        """
        if results_log.empty:
            logger.warning("Cannot plot price/bounds: results log is empty.")
            return

        required_cols = ['timestamp', 'price', 'lower_bound', 'upper_bound', 'rebalanced']
        if not all(col in results_log.columns for col in required_cols):
             missing = [col for col in required_cols if col not in results_log.columns]
             logger.error(f"Cannot plot price/bounds: results log missing required columns: {missing}")
             return

        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(results_log['timestamp']):
             try:
                 results_log['timestamp'] = pd.to_datetime(results_log['timestamp'])
             except Exception as e:
                  logger.error(f"Failed to convert timestamp column for plotting: {e}")
                  return

        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot price
        ax.plot(results_log['timestamp'], results_log['price'], label='Price', color='black', linewidth=1.5, zorder=3)

        # Plot position bounds over time
        ax.plot(results_log['timestamp'], results_log['lower_bound'], label='Lower Bound', color='red', linestyle='--', linewidth=1, zorder=2)
        ax.plot(results_log['timestamp'], results_log['upper_bound'], label='Upper Bound', color='green', linestyle='--', linewidth=1, zorder=2)

        # Fill between bounds to show the position range
        ax.fill_between(results_log['timestamp'], results_log['lower_bound'], results_log['upper_bound'],
                        color='blue', alpha=0.15, label='Position Range', zorder=1)

        # Mark rebalance events
        rebalance_points = results_log[results_log['rebalanced']]
        if not rebalance_points.empty:
            ax.scatter(rebalance_points['timestamp'], rebalance_points['price'],
                       marker='o', color='orange', s=60, zorder=5, label='Rebalance Event', edgecolors='black')

        # Formatting
        title = self._generate_plot_title(parameters)
        ax.set_title(f"Price vs Position Bounds\n{title}", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='best')
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5) # Fainter minor grid

        # Improve date formatting on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10)) # Adjust number of ticks
        fig.autofmt_xdate() # Rotate dates for better readability

        # Set y-axis limits dynamically based on data range, adding some padding
        min_val = results_log[['price', 'lower_bound']].min().min()
        max_val = results_log[['price', 'upper_bound']].max().max()
        padding = (max_val - min_val) * 0.05 # 5% padding
        ax.set_ylim(min_val - padding, max_val + padding)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        if output_file:
            filepath = os.path.join(self.output_dir, output_file)
            try:
                plt.savefig(filepath, dpi=150) # Increase DPI for better quality
                logger.info(f"Price and bounds plot saved to {filepath}")
            except Exception as e:
                 logger.error(f"Failed to save plot to {filepath}: {e}")
            finally:
                 plt.close(fig) # Close the figure after saving or error
        else:
            plt.show()
            plt.close(fig) # Close after showing

    def plot_performance_metrics(
        self,
        results_log: pd.DataFrame,
        parameters: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> None:
        """
        Plots key performance metrics over time (cumulative fees, position value).

        Args:
            results_log (pd.DataFrame): DataFrame containing the daily log from the backtest.
                                        Required columns: timestamp, cumulative_fees, position_value.
            parameters (Dict): Dictionary of backtest parameters for context in the title.
            output_file (Optional[str]): Path relative to output_dir to save the plot. If None, plot is shown.
        """
        if results_log.empty:
            logger.warning("Cannot plot performance: results log is empty.")
            return

        required_cols = ['timestamp', 'cumulative_fees', 'position_value']
        if not all(col in results_log.columns for col in required_cols):
             missing = [col for col in required_cols if col not in results_log.columns]
             logger.error(f"Cannot plot performance: results log missing required columns: {missing}")
             return

        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(results_log['timestamp']):
             try:
                 results_log['timestamp'] = pd.to_datetime(results_log['timestamp'])
             except Exception as e:
                  logger.error(f"Failed to convert timestamp column for plotting: {e}")
                  return

        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Plot Position Value on primary y-axis (ax1)
        color1 = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Position Value (USD)', color=color1)
        ax1.plot(results_log['timestamp'], results_log['position_value'], color=color1, label='Position Value (Assets)', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.6, alpha=0.7) # Grid for primary axis

        # Create secondary y-axis (ax2) sharing the same x-axis
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Cumulative Fees Earned (USD)', color=color2)
        ax2.plot(results_log['timestamp'], results_log['cumulative_fees'], color=color2, linestyle='--', label='Cumulative Fees', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Optional: Plot total value (Position Value + Cumulative Fees) on primary axis
        total_value = results_log['position_value'] + results_log['cumulative_fees']
        ax1.plot(results_log['timestamp'], total_value, color='tab:purple', linestyle=':', label='Total Value (Assets + Fees)', linewidth=1.5)


        # Formatting
        title = self._generate_plot_title(parameters)
        ax1.set_title(f"Performance Over Time\n{title}", fontsize=12)

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        # Improve date formatting on x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        if output_file:
            filepath = os.path.join(self.output_dir, output_file)
            try:
                plt.savefig(filepath, dpi=150)
                logger.info(f"Performance plot saved to {filepath}")
            except Exception as e:
                 logger.error(f"Failed to save plot to {filepath}: {e}")
            finally:
                 plt.close(fig) # Close the figure
        else:
            plt.show()
            plt.close(fig) # Close after showing

    # --- Methods below might be used if optimization features are added later ---
    # (Keep them commented out for now)
    # def plot_parameter_comparison(...)
    # def plot_heatmap(...)
    # def generate_summary_report(...)