# Tests

Simple tests for the class-based architecture using minimal fixtures.

## Running Tests

```bash
# Run all tests
uv run python -m tests.test_basic
uv run python -m tests.test_aggregations

# Or run linting (includes type checking)
make lint
```

## Test Structure

### Test Logs (`tests/9999_1P_2D_20251110_120000/`)

Minimal fake benchmark run based on real job structure:

- **`9999.json`** - Job metadata in {jobid}.json format
- **`test-node-cn01_prefill_w0.err`** - Prefill node logs with DP0, TP0/1/2
- **`test-node-cn02_decode_w0.err`** - Decode node logs with DP0, TP0/1/2
- **`test-node-cn03_decode_w1.err`** - Decode node logs with DP1, TP0/1/2
- **`vllm_isl_512_osl_256/*.json`** - Benchmark result files

**Key features for testing:**

- Different DP ranks (DP0, DP1) - tests DP grouping
- Different TP ranks (TP0, TP1, TP2) - tests TP averaging
- Mix of prefill and decode nodes - tests filtering
- Cached tokens in some batches - tests cache hit rate calculation

### Test Files

**`test_basic.py`** - Core functionality tests:

- `test_run_loader()` - Tests RunLoader can load from JSON
- `test_node_analyzer()` - Tests NodeAnalyzer can parse logs
- `test_node_count()` - Tests node counting

**`test_aggregations.py`** - Aggregation/grouping tests:

- `test_group_by_dp()` - Tests DP rank grouping (averages TP0/1/2 within each DP)
- `test_aggregate_all()` - Tests full aggregation (averages all nodes)
- `test_batch_metrics_calculations()` - Tests cache hit rate property

## What's Tested

✅ **RunLoader:**

- Loading runs from JSON metadata
- Parsing profiler benchmark results
- Run metadata properties (formatted_date, total_gpus, etc.)

✅ **NodeAnalyzer:**

- Parsing .err log files
- Converting to NodeMetrics objects
- Filtering prefill/decode nodes
- Node counting

✅ **Models:**

- BatchMetrics with different DP/TP/EP values
- Cache hit rate calculation
- Node properties (is_prefill, is_decode)

✅ **Aggregations:**

- Group by DP rank (averages TP workers)
- Aggregate all nodes (single averaged line)
- Proper averaging across different DP/TP combinations

## Design Philosophy

- **Minimal fixtures** - Only essential data, no bloat
- **No test helpers** - Direct testing, easy to understand
- **Fast execution** - Tests run in < 1 second
- **Type-safe** - All tests pass type checking
