"""
LP Optimizer CLI - Main Entry Point

Command-line interface for backtesting vfat.io LP strategies on DEX pools
using historical data fetched directly from subgraphs.
"""

import logging
import argparse
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from types import SimpleNamespace

# Use absolute imports assuming 'src' is the root package or accessible in PYTHONPATH
from src.simulation.backtest_engine import BacktestEngine
from src.simulation.result_processor import ResultProcessor
from src.data.data_loader import load_data
from src.simulation import config as cfg
from src.utils.helpers import ensure_dir_exists, parse_date
from src.api.subgraph_client import get_client

# --- Configuration ---
# Load config early to access defaults and API settings
config = cfg.get_config()

# Configure logging
log_level = logging.INFO # Default level
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__) # Logger for this module
# Silence overly verbose libraries if needed
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("python_graphql_client").setLevel(logging.WARNING)

# --- Exchange Configuration ---
# Define supported exchanges with display symbols
config["exchanges"] = {
    "aerodrome": {"symbol": "1", "display": "Aerodrome (Base)"},
    "shadow": {"symbol": "2", "display": "Shadow (Sonic)"}
}

# --- Helper Functions ---

def get_pools_for_exchange(exchange_name):
    """Fetches the top pools for a given exchange.
    
    Args:
        exchange_name (str): The exchange identifier (e.g., 'aerodrome', 'shadow')
        
    Returns:
        list: A list of pool objects with pool details or empty list if failed.
    """
    logger.info(f"Fetching top pools for {exchange_name}")
    print(f"\n⏳ Fetching pools from {exchange_name.capitalize()}...")
    
    client = get_client(exchange_name)
    if not client:
        logger.error(f"Failed to create client for {exchange_name}")
        return []
    
    try:
        # Fetch top 20 pools by TVL
        pools = client.get_top_pools(limit=20)
        if not pools:
            logger.warning(f"No pools returned for {exchange_name}")
            return []
        
        logger.info(f"Retrieved {len(pools)} pools for {exchange_name}")
        return pools
    except Exception as e:
        logger.error(f"Error fetching pools for {exchange_name}: {e}")
        return []

# --- Command Functions ---

def run_single_backtest(args):
    """Runs a single vfat strategy backtest based on CLI arguments."""
    # Initial setup information
    print("\n" + "="*60)
    print(f"  Backtesting {args.width}% vfat.io Strategy on {args.exchange.upper()}")
    print(f"  Pool: {args.pool_address}")
    print(f"  Date Range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    print("="*60)
    
    logger.info(f"Initiating backtest for {args.exchange.upper()} pool: {args.pool_address}")
    logger.info(f"Date Range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Strategy Params: Width={args.width}%, RebalanceBuffer={args.rebalance_buffer}%, CutoffBuffer={args.cutoff_buffer}%")
    logger.info(f"Investment: ${args.investment:.2f}")
    logger.info(f"Simulate Tx Costs: {not args.no_tx_costs}")
    if not args.no_tx_costs:
         # Use CLI args if provided, otherwise they are None and engine uses config defaults
         gas_override = args.gas_cost
         slippage_override = args.slippage_pct
         logger.info(f"Tx Costs: Gas=${gas_override if gas_override is not None else 'config_default'}, Slippage={slippage_override if slippage_override is not None else 'config_default'}%")

    # --- 1. Load Data ---
    print("\n⏳ Fetching pool details and historical data...")
    logger.info("Loading data from subgraph...")
    try:
        pool_details, historical_data = load_data(
            exchange_name=args.exchange,
            pool_address=args.pool_address,
            start_date=args.start_date,
            end_date=args.end_date
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to load data due to an unexpected error: {e}")
        sys.exit(1) # Exit with error code

    # Validate loaded data
    if pool_details is None:
        # Data loader logs specific errors, provide a user-friendly message
        logger.error("Failed to load pool details. Cannot proceed without essential info like fee tier.")
        print("\n❌ ERROR: Could not load pool details for the specified address.")
        print("Please verify the pool address and exchange, and check network/API connectivity.")
        sys.exit(1)

    if historical_data is None:
        logger.error("Failed to load historical data (returned None).")
        print("\n❌ ERROR: Could not load historical data. Subgraph query might have failed.")
        print("Please check the pool address, date range, and network/API connectivity.")
        sys.exit(1)

    if historical_data.empty:
        logger.error("Historical data is empty for the specified range.")
        print("\n❌ ERROR: No historical data found for the specified pool and date range.")
        print("Please check the pool address and date range. The pool might be new or data unavailable.")
        sys.exit(1)

    # Add exchange name to pool_details if not present (useful for parameters log)
    if 'exchange' not in pool_details:
         pool_details['exchange'] = args.exchange

    print(f"✅ Data loaded successfully: {len(historical_data)} historical data points.")
    pool_name = f"{pool_details.get('token0', {}).get('symbol', 'Token0')}/{pool_details.get('token1', {}).get('symbol', 'Token1')}"
    print(f"   Pool: {pool_name}")
    print(f"   Fee Tier: {pool_details.get('feeTier', 'Unknown')} bps")
    logger.info(f"Loaded {len(historical_data)} historical data points.")

    # --- 2. Initialize Backtest Engine ---
    print("\n⏳ Initializing backtest engine...")
    logger.info("Initializing backtest engine...")
    try:
        engine = BacktestEngine(
            pool_details=pool_details,
            historical_data=historical_data,
            config=config,
            gas_cost_usd=args.gas_cost, # Pass overrides (can be None)
            slippage_pct=args.slippage_pct # Pass overrides (can be None)
        )
    except ValueError as e:
        logger.error(f"Failed to initialize BacktestEngine: {e}")
        print(f"\n❌ ERROR: Could not initialize backtest engine: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during engine initialization: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to initialize engine due to an unexpected error: {e}")
        sys.exit(1)

    # --- 3. Run Backtest ---
    print("\n⏳ Running backtest simulation...")
    print(f"   - Width: {args.width}%")
    print(f"   - Rebalance Buffer: {args.rebalance_buffer}")
    print(f"   - Cutoff Buffer: {args.cutoff_buffer}")
    print(f"   - Initial Investment: ${args.investment:,.2f}")
    print(f"   - Transaction Costs: {'Included' if not args.no_tx_costs else 'Excluded'}")
    
    logger.info("Starting vfat backtest simulation...")
    try:
        result = engine.run_vfat_backtest(
            initial_width_pct=args.width,
            rebalance_buffer_pct=args.rebalance_buffer,
            cutoff_buffer_pct=args.cutoff_buffer,
            initial_investment=args.investment,
            simulate_tx_costs=not args.no_tx_costs
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during backtest execution: {e}", exc_info=True)
        print(f"\n❌ ERROR: Backtest simulation failed unexpectedly: {e}")
        sys.exit(1)


    # --- 4. Process Results ---
    if not result or "metrics" not in result or "parameters" not in result:
        logger.error("Backtest failed to produce valid results.")
        print("\n❌ ERROR: Backtest simulation did not return expected results.")
        sys.exit(1)
    if "error" in result:
         logger.error(f"Backtest simulation returned an error: {result['error']}")
         print(f"\n❌ ERROR: Backtest simulation failed: {result['error']}")
         sys.exit(1)


    print("\n✅ Backtest simulation complete.")
    logger.info("Processing and displaying results...")

    output_dir = config.get("output_dir", "./output")
    ensure_dir_exists(output_dir) # Ensure output dir exists
    processor = ResultProcessor(output_dir=output_dir)

    # Print summary table to console
    print("\n" + "="*40)
    print("      Backtest Simulation Results")
    print("="*40)
    processor.print_summary_metrics(result["metrics"])
    print("="*40)


    # --- 5. Save Outputs (Plots, Logs) ---
    # Generate a base filename for outputs
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize pool address for filename
    safe_pool_addr = args.pool_address.replace('0x', '')[:12] # Shortened address
    token0 = result['parameters'].get('token0', 'T0')
    token1 = result['parameters'].get('token1', 'T1')
    filename_base = (
        f"{args.exchange}_{token0}_{token1}_{safe_pool_addr}_"
        f"w{args.width}_rb{args.rebalance_buffer}_co{args.cutoff_buffer}_"
        f"{timestamp_str}"
    ).replace('%', 'pct') # Basic filename generation


    # Generate plots if requested
    if args.plot:
        print("\n⏳ Generating plots...")
        logger.info("Generating plots...")
        try:
            plot_price_filename = f"{filename_base}_price_bounds.png"
            processor.plot_price_and_bounds(
                result["results_log"],
                result["parameters"], # Pass parameters for title context
                output_file=plot_price_filename
            )

            plot_perf_filename = f"{filename_base}_performance.png"
            processor.plot_performance_metrics(
                 result["results_log"],
                 result["parameters"], # Pass parameters for title context
                 output_file=plot_perf_filename
            )
            print(f"\n✅ Plots saved to directory: {os.path.abspath(output_dir)}")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}", exc_info=True)
            print(f"\n⚠️ WARNING: Failed to generate plots: {e}")

    # Save detailed log if requested
    if args.save_log:
        print("\n⏳ Saving detailed results log...")
        log_filename = os.path.join(output_dir, f"{filename_base}_details.csv")
        try:
            # Ensure results_log is a DataFrame before saving
            if isinstance(result.get("results_log"), pd.DataFrame):
                result["results_log"].to_csv(log_filename, index=False)
                print(f"✅ Detailed results log saved to: {log_filename}")
            else:
                 logger.error("results_log is not a DataFrame, cannot save CSV.")
                 print("\n⚠️ WARNING: Could not save detailed log, results format unexpected.")
        except Exception as e:
            logger.error(f"Failed to save detailed log to {log_filename}: {e}", exc_info=True)
            print(f"\n⚠️ WARNING: Failed to save detailed log: {e}")

    print("\n" + "="*60)
    print("  Backtest completed successfully!")
    print("="*60)


def run_custom_mode(args=None):
    """Runs the tool in custom interactive mode."""
    args = SimpleNamespace()  # Create a namespace to mimic CLI arg structure
    
    # Load exchanges and pools
    print("\n\n=== Backtest Configuration ===")
    print("\nTip: Press ENTER to select default options shown in [brackets]")
    
    # Step 1: Choose Exchange
    exchanges = list(config["exchanges"].keys())
    print("\nAvailable Exchanges (enter the number):")
    for i, exchange in enumerate(exchanges, start=1):
        display_name = config["exchanges"][exchange].get("display", exchange.capitalize())
        print(f"{i}. {display_name}")
    
    # Get exchange selection
    while True:
        try:
            exchange_input = input("\nEnter exchange number (1-" + str(len(exchanges)) + "): ")
            exchange_idx = int(exchange_input) - 1
            if 0 <= exchange_idx < len(exchanges):
                args.exchange = exchanges[exchange_idx]
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(exchanges)}.")
        except ValueError:
            print("Please enter a valid number, not a letter.")
    
    # Step 2: Choose pool from available pools or enter manually
    pool_selection_complete = False
    manual_entry_attempts = 0
    
    while not pool_selection_complete:
        pool_data = get_pools_for_exchange(args.exchange)
        
        if not pool_data or len(pool_data) == 0:
            print(f"\n⚠️ Could not fetch pools for {args.exchange}. This could be due to network issues or API limitations.")
            print("Options:")
            print("1. Try again")
            print("2. Enter pool address manually")
            print("3. Go back to exchange selection")
            
            while True:
                option = input("\nChoose an option (1-3): ")
                if option == "1":
                    # Continue the loop to try fetching pools again
                    break
                elif option == "2":
                    # Enter pool address manually
                    while True:
                        manual_pool = input("\nEnter pool address (0x...): ").strip()
                        if manual_pool.startswith("0x") and len(manual_pool) >= 42:
                            args.pool_address = manual_pool
                            pool_selection_complete = True
                            break
                        else:
                            manual_entry_attempts += 1
                            print("Invalid pool address format. Address should start with 0x and be at least 42 characters.")
                            if manual_entry_attempts >= 3:
                                print("\n⚠️ Multiple failed attempts. Please verify the pool exists on the selected exchange.")
                                manual_entry_attempts = 0
                                
                    break
                elif option == "3":
                    # Go back to exchange selection
                    while True:
                        try:
                            exchange_input = input("\nEnter exchange number (1-" + str(len(exchanges)) + "): ")
                            exchange_idx = int(exchange_input) - 1
                            if 0 <= exchange_idx < len(exchanges):
                                args.exchange = exchanges[exchange_idx]
                                break
                            else:
                                print(f"Invalid selection. Please enter a number between 1 and {len(exchanges)}.")
                        except ValueError:
                            print("Please enter a valid number, not a letter.")
                    break
                else:
                    print("Invalid option. Please enter 1, 2, or 3.")
        else:
            # Display available pools
            print(f"\nAvailable {args.exchange.capitalize()} Pools (enter the number):")
            for i, pool in enumerate(pool_data, start=1):
                pool_name = f"{pool['token0']['symbol']}/{pool['token1']['symbol']}"
                print(f"{i}. {pool_name} (Fee: {pool['feeTier']} bps)")
            
            # Get pool selection
            while True:
                try:
                    pool_input = input(f"\nEnter pool number (1-{len(pool_data)}) or 'a' to enter address manually: ")
                    
                    if pool_input.lower() == 'a':
                        manual_pool = input("\nEnter pool address (0x...): ").strip()
                        if manual_pool.startswith("0x") and len(manual_pool) >= 42:
                            args.pool_address = manual_pool
                            pool_selection_complete = True
                            break
                        else:
                            print("Invalid pool address format. Address should start with 0x and be at least 42 characters.")
                            continue
                    
                    pool_idx = int(pool_input) - 1
                    if 0 <= pool_idx < len(pool_data):
                        selected_pool = pool_data[pool_idx]
                        args.pool_address = selected_pool["id"]
                        pool_selection_complete = True
                        break
                    else:
                        print(f"Invalid selection. Please enter a number between 1 and {len(pool_data)}.")
                except ValueError:
                    if pool_input.lower() != 'a':
                        print("Please enter a valid number or 'a' to enter address manually.")
                    else:
                        continue
            break  # Exit the pool selection loop

    # If we've gotten here with a manual pool address, create a minimal selected_pool object
    if "selected_pool" not in locals():
        selected_pool = {"id": args.pool_address, "token0": {"symbol": "?"}, "token1": {"symbol": "?"}}
        print(f"\nUsing manually entered pool: {args.pool_address}")
        print("Note: Pool details will be fetched during the backtest")
    
    # Step 3: Choose date range
    # Default to 30 days lookback
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Get user input for dates or use defaults
    print("\nSelect Date Range:")
    date_format = "%Y-%m-%d"
    
    start_date_input = input(f"Start Date [{start_date.strftime(date_format)}]: ")
    if start_date_input:
        try:
            args.start_date = datetime.strptime(start_date_input, date_format)
        except ValueError:
            print(f"Invalid date format. Using default: {start_date.strftime(date_format)}")
            args.start_date = start_date
    else:
        args.start_date = start_date
    
    end_date_input = input(f"End Date [{end_date.strftime(date_format)}]: ")
    if end_date_input:
        try:
            args.end_date = datetime.strptime(end_date_input, date_format)
        except ValueError:
            print(f"Invalid date format. Using default: {end_date.strftime(date_format)}")
            args.end_date = end_date
    else:
        args.end_date = end_date
    
    # Step 4: Strategy parameters
    print("\nStrategy Parameters:")
    
    # Position width (default 0.5%)
    width_input = input("Position Width % [0.5]: ")
    args.width = float(width_input) if width_input else 0.5
    
    # Rebalance buffer (default 1.0%)
    rebalance_input = input("Rebalance Buffer % [1.0]: ")
    args.rebalance_buffer = float(rebalance_input) if rebalance_input else 1.0
    
    # Cutoff buffer (default same as rebalance buffer)
    default_cutoff = args.rebalance_buffer
    cutoff_input = input(f"Cutoff Buffer % [{default_cutoff}]: ")
    args.cutoff_buffer = float(cutoff_input) if cutoff_input else default_cutoff
    
    # Step 5: Investment amount
    investment_input = input("Investment Amount USD [10000]: ")
    args.investment = float(investment_input) if investment_input else 10000
    
    # Step 6: Transaction cost simulation
    tx_cost_input = input("Include Transaction Costs? (y/n) [y]: ")
    args.no_tx_costs = tx_cost_input.lower() in ['n', 'no']
    
    # If simulating tx costs, get gas and slippage defaults or overrides
    if not args.no_tx_costs:
        # Get defaults from config for display
        default_gas = config.get("gas_cost_usd", 5)
        default_slippage = config.get("slippage_pct", 0.05)
        
        gas_input = input(f"Gas Cost USD (per tx) [{default_gas}]: ")
        args.gas_cost = float(gas_input) if gas_input else None  # None means use config default
        
        slippage_input = input(f"Slippage % [{default_slippage}]: ")
        args.slippage_pct = float(slippage_input) if slippage_input else None  # None means use config default
    else:
        # Not using tx costs, set both to None
        args.gas_cost = None
        args.slippage_pct = None
    
    # Step 7: Output options
    plot_input = input("Generate Price and Performance Plots? (y/n) [y]: ")
    args.plot = not (plot_input.lower() in ['n', 'no'])
    
    save_log_input = input("Save Detailed Results Log? (y/n) [n]: ")
    args.save_log = save_log_input.lower() in ['y', 'yes']
    
    # Display configuration summary
    print("\n=== Backtest Configuration Summary ===")
    print(f"Exchange: {args.exchange.capitalize()}")
    pool_name = f"{selected_pool['token0']['symbol']}/{selected_pool['token1']['symbol']}"
    print(f"Pool: {pool_name}")
    print(f"Date Range: {args.start_date.strftime(date_format)} to {args.end_date.strftime(date_format)}")
    print(f"Strategy: vfat.io with {args.width}% width, {args.rebalance_buffer}% rebalance buffer")
    print(f"Investment: ${args.investment:,.2f}")
    
    tx_costs_status = "Included" if not args.no_tx_costs else "Excluded"
    tx_costs_details = ""
    if not args.no_tx_costs:
        gas_display = args.gas_cost if args.gas_cost is not None else f"{default_gas} (default)"
        slippage_display = args.slippage_pct if args.slippage_pct is not None else f"{default_slippage} (default)"
        tx_costs_details = f" (Gas=${gas_display}, Slippage={slippage_display}%)"
    
    print(f"Transaction Costs: {tx_costs_status}{tx_costs_details}")
    
    # Confirm execution
    confirm = input("\nProceed with backtest? (y/n) [y]: ")
    if confirm.lower() in ['n', 'no']:
        print("Backtest cancelled.")
        return
    
    # Run the backtest with configured parameters
    run_single_backtest(args)


def run_quick_backtest(args):
    """
    Run a backtest with minimal parameters and sensible defaults.
    This is a streamlined version that requires minimal input.
    """
    # Calculate date ranges from days argument
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)
    
    # Convert to datetime objects
    end_date = datetime.combine(end_date, datetime.min.time())
    start_date = datetime.combine(start_date, datetime.min.time())
    
    # Execute the backtest with quick options
    logger.info(f"Quick backtest for {args.exchange} pool {args.pool_address}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({args.days} days)")
    logger.info(f"Strategy: Width {args.width}%, Rebalance Buffer {args.rebalance_buffer}, Cutoff Buffer {args.cutoff_buffer}")
    
    # Create a new args object with all the required properties for run_single_backtest
    class QuickArgs:
        pass
    
    quick_args = QuickArgs()
    quick_args.exchange = args.exchange
    quick_args.pool_address = args.pool_address
    quick_args.start_date = start_date
    quick_args.end_date = end_date
    quick_args.width = args.width
    quick_args.rebalance_buffer = args.rebalance_buffer
    quick_args.cutoff_buffer = args.cutoff_buffer
    quick_args.investment = args.investment
    quick_args.no_tx_costs = args.no_tx
    quick_args.gas_cost = None  # Use default from config
    quick_args.slippage_pct = None  # Use default from config
    quick_args.plot = args.plot
    quick_args.save_log = args.log
    quick_args.verbose = False
    
    # Run the backtest with our configured args
    run_single_backtest(quick_args)


# --- Main CLI Setup ---

def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description="LP Optimizer CLI: Backtest vfat.io strategies for DEX pools using real historical data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # --- Subparsers for commands ---
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # --- Backtest Command Arguments ---
    backtest_parser = subparsers.add_parser("backtest", help="Run a single vfat strategy backtest.",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    backtest_parser.add_argument("--exchange", type=str, required=True, choices=['aerodrome', 'shadow'],
                                 help="Required: The exchange name ('aerodrome' on Base or 'shadow' on Sonic).")
    backtest_parser.add_argument("--pool-address", type=str, required=True,
                                 help="Required: The address of the liquidity pool.")
    backtest_parser.add_argument("--width", type=float, required=True,
                                 help="Required: Initial position width percentage relative to entry price (e.g., 1.0 for +/- 0.5%).")
    backtest_parser.add_argument("--rebalance-buffer", type=float, required=True,
                                 help="Required: Rebalance trigger distance as a percentage of width (e.g., 0.1 for 10% of width).")
    backtest_parser.add_argument("--cutoff-buffer", type=float, required=True,
                                 help="Required: Rebalance prevention distance as a percentage of width (e.g., 0.5 for 50% of width). Must be >= rebalance-buffer or 0 to disable cutoff.")

    # Optional arguments with defaults
    backtest_parser.add_argument("--start-date", type=str, default=None,
                                 help="Start date for backtest (YYYY-MM-DD). Default: 90 days before end date.")
    backtest_parser.add_argument("--end-date", type=str, default=None,
                                 help="End date for backtest (YYYY-MM-DD). Default: yesterday.")
    backtest_parser.add_argument("--investment", type=float,
                                 default=config.get("backtest_defaults", {}).get("initial_investment", 10000.0),
                                 help="Initial investment amount in USD.")

    # Transaction cost arguments (optional overrides for config defaults)
    tx_group = backtest_parser.add_argument_group('Transaction Cost Overrides (Optional)')
    tx_group.add_argument("--no-tx-costs", action="store_true",
                          help="Disable simulation of gas and slippage costs entirely.")
    tx_group.add_argument("--gas-cost", type=float, default=None,
                          help=f"Override the gas cost in USD per rebalance transaction. Default from config: ${config.get('transaction_costs', {}).get('rebalance_gas_usd', 'N/A')}")
    tx_group.add_argument("--slippage-pct", type=float, default=None,
                           help=f"Override the slippage percentage per rebalance swap. E.g., 0.1 for 0.1%. Default from config: {config.get('transaction_costs', {}).get('slippage_percentage', 'N/A')}%")


    # Output arguments
    output_group = backtest_parser.add_argument_group('Output Options')
    output_group.add_argument("--plot", action="store_true",
                              help="Generate and save result plots (price/bounds, performance) to the output directory.")
    output_group.add_argument("--save-log", action="store_true",
                              help="Save the detailed daily backtest log to a CSV file in the output directory.")
    output_group.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG level logging for detailed output.")

    # Set the function to call for the 'backtest' command
    backtest_parser.set_defaults(func=run_single_backtest)
    
    # --- Quick Command (Streamlined one-liner) ---
    quick_parser = subparsers.add_parser("quick", 
                                        help="Run a backtest with minimal required parameters and sensible defaults.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required args for quick command
    quick_parser.add_argument("exchange", type=str, choices=['aerodrome', 'shadow'],
                             help="Exchange name ('aerodrome' on Base or 'shadow' on Sonic).")
    quick_parser.add_argument("pool_address", type=str,
                             help="Address of the liquidity pool.")
    quick_parser.add_argument("width", type=float, nargs='?', default=1.0,
                             help="Position width percentage (default: 1.0).")
    
    # Optional quick command args with reasonable defaults
    quick_parser.add_argument("--rb", type=float, dest="rebalance_buffer", default=0.2,
                             help="Rebalance buffer (default: 0.2).")
    quick_parser.add_argument("--cb", type=float, dest="cutoff_buffer", default=0.5,
                             help="Cutoff buffer (default: 0.5). Set to 0 to disable cutoff.")
    quick_parser.add_argument("--investment", "-i", type=float, default=10000.0,
                             help="Investment amount in USD (default: 10000).")
    quick_parser.add_argument("--days", "-d", type=int, default=90,
                             help="Days to backtest (default: 90).")
    quick_parser.add_argument("--plot", "-p", action="store_true", default=True,
                             help="Generate plots (default: enabled).")
    quick_parser.add_argument("--log", "-l", action="store_true", default=True,
                             help="Save detailed log (default: enabled).")
    quick_parser.add_argument("--no-tx", action="store_true", 
                             help="Disable transaction costs (default: costs enabled).")
    
    quick_parser.set_defaults(func=run_quick_backtest)
    
    # --- Wizard Command (Interactive Mode) ---
    custom_parser = subparsers.add_parser("custom", 
                                         help="Interactive custom mode for guided backtest setup.",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # No required arguments for custom mode - it's interactive
    custom_parser.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG level logging for detailed output.")
    custom_parser.set_defaults(func=run_custom_mode)

    # Parse arguments
    args = parser.parse_args()

    # --- Post-parsing Setup ---

    # Set logging level based on verbosity flag
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
             handler.setFormatter(logging.Formatter(log_format)) # Ensure format consistency
        logger.info("Verbose logging enabled (DEBUG level).")
    else:
        logging.getLogger().setLevel(logging.INFO)
        # Update formatter for existing handlers if needed (though basicConfig usually handles it)
        for handler in logging.getLogger().handlers:
             handler.setFormatter(logging.Formatter(log_format))

    # Validate and parse dates for backtest command
    if args.command == "backtest":
        try:
            if args.end_date is None:
                # Default to yesterday to ensure the last full day's data is likely available
                args.end_date = datetime.now().date() - timedelta(days=1)
            else:
                args.end_date = parse_date(args.end_date).date() # Use only the date part

            if args.start_date is None:
                default_days = config.get("backtest_defaults", {}).get("days", 90)
                args.start_date = args.end_date - timedelta(days=default_days)
            else:
                 args.start_date = parse_date(args.start_date).date() # Use only the date part

            # Ensure start_date is before end_date
            if args.start_date >= args.end_date:
                 parser.error(f"Start date ({args.start_date}) must be strictly before end date ({args.end_date}).")

            # Convert dates to datetime objects for functions that need them
            # Set time to start of day for start_date, end of day for end_date?
            # Let's use start of day for both for simplicity with daily data.
            args.start_date = datetime.combine(args.start_date, datetime.min.time())
            args.end_date = datetime.combine(args.end_date, datetime.min.time())


        except ValueError as e:
            parser.error(f"Invalid date format: {e}. Please use YYYY-MM-DD.")

        # Validate buffer percentages
        if hasattr(args, 'rebalance_buffer') and hasattr(args, 'cutoff_buffer'):
             if args.cutoff_buffer < args.rebalance_buffer and args.cutoff_buffer != 0:
                  parser.error(f"Cutoff buffer ({args.cutoff_buffer}%) must be greater than or equal to rebalance buffer ({args.rebalance_buffer}%) or set to 0 to disable cutoff.")


    # Execute the function associated with the chosen command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # Should not happen if command is required, but good practice
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 