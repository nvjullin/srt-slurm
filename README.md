# SRT Slurm Benchmark Dashboard

Interactive Streamlit dashboard for visualizing and analyzing end to end sglang benchmarks run on SLURM clusters.

> [!NOTE]
> You must use the [slurm jobs folder](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/slurm_jobs) in the dynamo repository to run the job so that this benchmarking tools can analyze it

## Quick Start

```bash
./run_dashboard.sh
```

The dashboard will open at http://localhost:8501 and scan the current directory for benchmark runs.

## What It Does

**Pareto Analysis** - Compare throughput efficiency (TPS/GPU) vs per-user throughput (TPS/User) across configurations

**Latency Breakdown** - Visualize TTFT, TPOT, and ITL metrics as concurrency increases

**Config Comparison** - View deployment settings (TP/DP) and hardware specs side-by-side

**Data Export** - Sort, filter, and export metrics to CSV

## Key Metrics

- **Output TPS/GPU** - Throughput per GPU (higher = more efficient)
- **Output TPS/User** - Throughput per concurrent user (higher = better responsiveness)
- **TTFT** - Time to first token (lower = faster start)
- **TPOT** - Time per output token (lower = faster generation)
- **ITL** - Inter-token latency (lower = smoother streaming)

## Installation

**With uv (recommended):**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the dashboard (uv handles dependencies automatically)
./run_dashboard.sh
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Setup pre-commit hooks
pre-commit install

# Run linting and type checking
make lint

# Run tests
uv run python -m tests.test_basic
uv run python -m tests.test_aggregations
```

## Directory Structure

The app expects benchmark runs in subdirectories with:

- `{jobid}.json` - Metadata file with run configuration (required)
- `vllm_isl_*_osl_*/` containing `*.json` result files
- `*_config.json` files for node configurations
- `*_prefill_*.err` and `*_decode_*.err` files for node metrics

## Architecture

Uses a **class-based architecture** with type-safe models:

```python
from srtslurm import RunLoader, NodeAnalyzer

# Load benchmark runs from {jobid}.json files
loader = RunLoader(".")
runs = loader.load_all()  # Returns List[BenchmarkRun]

# Access typed data
run = runs[0]
print(f"{run.job_id}: {run.metadata.formatted_date}")
print(f"Topology: {run.metadata.prefill_nodes}P/{run.metadata.decode_nodes}D")
print(f"GPUs: {run.metadata.total_gpus}")

# Analyze node metrics from .err log files
analyzer = NodeAnalyzer()
nodes = analyzer.parse_run_logs(run.metadata.path)
prefill = analyzer.get_prefill_nodes(nodes)
decode = analyzer.get_decode_nodes(nodes)
```

See `CLASS_BASED_USAGE.md` for detailed API documentation and `LOG_STRUCTURE.md` for file formats.
