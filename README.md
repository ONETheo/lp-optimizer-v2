# LP Optimizer CLI (Python)

A Python command-line tool for backtesting liquidity provider (LP) strategies, specifically simulating the vfat.io rebalancing approach, for pools on DEXes like Aerodrome and Shadow Finance using real historical data from TheGraph subgraphs.

## Features

- Fetch historical pool data (price, TVL, volume, fees) directly from Aerodrome & Shadow subgraphs
- Backtest the vfat.io rebalancing strategy:
  - Configurable initial position width
  - Configurable rebalance buffer percentage
  - Configurable cutoff buffer percentage (can be set to 0 to disable cutoff)
  - Simulation of transaction costs (gas fees and slippage)
- Calculate performance metrics (APR, Net APR, Fees APR, Rebalance Count, Time in Position, etc.)
- Command-line interface with detailed parameter control
- Generate plots for visualizing backtest results
- Save detailed simulation logs for further analysis

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/lp-optimizer.git # Replace with your repo URL
   cd lp-optimizer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your TheGraph Gateway API key:
   ```dotenv
   SUBGRAPH_API_KEY=your_thegraph_gateway_api_key_here
   AERODROME_SUBGRAPH_ENDPOINT=https://gateway.thegraph.com/api/subgraphs/id/GENunSHWLBXm59mBSgPzQ8metBEp9YDfdqwFr91Av1UM
   SHADOW_SUBGRAPH_ENDPOINT=your_shadow_subgraph_endpoint_here
   SHADOW_BACKUP_SUBGRAPH_ENDPOINT=your_shadow_backup_endpoint_here  # Optional backup endpoint
   ```

## Usage

The main entry point is `src/main.py`. You can run it using `python -m src.main`.

### Running a Backtest

```bash
python -m src.main backtest \
    --exchange aerodrome \
    --pool-address 0x70aCDF2Ad0bf2402C957154f944c19Ef4e1cbAE1 \
    --width 0.5 \
    --rebalance-buffer 0.1 \
    --cutoff-buffer 0.5 \
    --investment 10000 \
    --plot \
    --save-log
```

### Interactive Custom Mode

For an interactive guided setup, use the custom mode:

```bash
python -m src.main custom
```

This mode offers a streamlined interface with helpful explanations ideal for new users. It will walk you through selecting an exchange, pool, date range, and strategy parameters with guided prompts. For convenience:

- Press ENTER to select default values where available
- Default slippage values are provided based on the selected exchange
- A configuration summary is displayed before execution for verification

### Quick Command Mode

For faster testing with minimal parameters:

```bash
python -m src.main quick aerodrome 0x70aCDF2Ad0bf2402C957154f944c19Ef4e1cbAE1 1.0
```

### Command Line Arguments

Use `python -m src.main --help` to see all available commands and options.

#### Required Arguments

- `--exchange`: The exchange name ('aerodrome' or 'shadow')
- `--pool-address`: The address of the liquidity pool
- `--width`: Initial position width (percentage, e.g., 0.5 for +/- 0.25%)
- `--rebalance-buffer`: Percentage outside width to trigger rebalance (e.g., 0.1 for 10% of width)
- `--cutoff-buffer`: Percentage outside width to prevent rebalance (e.g., 0.5 for 50% of width). Can be set to 0 to disable cutoff completely.

#### Optional Arguments

- `--start-date`: Start date for backtest (YYYY-MM-DD, default: 90 days ago)
- `--end-date`: End date for backtest (YYYY-MM-DD, default: yesterday)
- `--investment`: Initial investment amount (default: 10000)
- `--no-tx-costs`: Flag to disable transaction cost simulation
- `--gas-cost`: Override the default gas cost in USD per rebalance
- `--slippage-pct`: Override the default slippage percentage per rebalance
- `--plot`: Generate and save result plots (price/bounds, performance)
- `--save-log`: Save detailed daily simulation log to CSV
- `--verbose`: Enable detailed debug logging

## Transaction Cost Simulation

By default, the tool simulates the impact of transaction costs:
- Gas fees per rebalance (default: $2.00)
- Slippage during swaps (default: 0.05%)

These defaults can be overridden using the `--gas-cost` and `--slippage-pct` flags, or disabled entirely with `--no-tx-costs`.

## Output

The tool provides several outputs:
1. Console summary of backtest metrics
2. Optional plots saved to `./output/` (when using `--plot`)
   - Price and rebalance bounds visualization
   - Performance metrics over time
3. Optional detailed CSV log of daily simulation data (when using `--save-log`)

All output files use a naming convention that includes:
- Exchange name
- Token symbols
- Pool address (shortened)
- Strategy parameters
- Timestamp

## Project Structure

```
lp-optimizer/
├── data/                   # Directory for potential future data storage (currently unused)
├── output/                 # Default output for plots/reports
├── src/
│   ├── api/                # Subgraph API clients
│   │   ├── __init__.py
│   │   ├── subgraph_client.py
│   │   ├── aerodrome/      # Aerodrome-specific client code
│   │   └── shadow/         # Shadow-specific client code
│   ├── data/               # Data loading module
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── simulation/         # Backtesting engine and results
│   │   ├── __init__.py
│   │   ├── backtest_engine.py
│   │   ├── config.py
│   │   └── result_processor.py
│   ├── utils/              # Helper utilities
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── __init__.py
│   └── main.py             # CLI entry point
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore configuration
├── requirements.txt        # Python dependencies
└── README.md
```

**Note:** The `node_modules` directory is not required in the repository and has been added to `.gitignore`. It contains JavaScript dependencies used by the TheGraph clients and will be generated automatically when needed.

## Advanced Configuration

You can customize default settings by:
1. Modifying environment variables in the `.env` file
2. Creating a custom config file (e.g., `config/custom_config.yaml`)

## License

MIT License