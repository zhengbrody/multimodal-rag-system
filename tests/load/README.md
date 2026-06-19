# Load Testing with JMeter

## Prerequisites

- Apache JMeter 5.5+ installed
- Target API server running

## Quick Start

```bash
# Run with defaults (localhost:8000)
./run_load_test.sh

# Run against a specific host
./run_load_test.sh my-api-host.example.com 8000
```

## Test Plan Overview

The test plan (`test_plan.jmx`) includes three thread groups:

1. **Health Check** - 100 threads over 60s ramp-up, 300s duration. Validates `/health` returns 200.
2. **Question Answering** - 500 threads over 120s ramp-up, 300s duration. Sends POST `/ask` with randomized questions from `questions.csv`.
3. **Concurrent Load** - 500 threads over 60s ramp-up, 180s duration. Mixed traffic: 70% `/ask`, 20% `/health`, 10% `/metrics`.

## Results

After a test run, results are written to `tests/load/results/`:

- `results.jtl` - Raw JMeter results
- `report/` - HTML dashboard report (open `report/index.html` in a browser)

## Customization

- Edit `questions.csv` to change the set of test questions.
- Adjust thread counts, ramp-up times, and durations in `test_plan.jmx` or via JMeter GUI.
