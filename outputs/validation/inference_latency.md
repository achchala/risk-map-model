# Inference Latency & Size Profiling

## Results

| Metric | Value |
|---|---|
| Single-segment prediction (median, n=200) | 14.361 ms |
| Batch prediction total (65,133 segments) | 232.3 ms |
| Batch prediction per segment | 0.0036 ms |
| Model file size | 0.88 MB |

## Interpretation
At **0.0036 ms/segment**, scoring all 65,133 road segments takes 232.3 ms —
well within the sub-second budget required for real-time routing API responses.
A single route query (1 segment lookup) takes 14.361 ms.

> **Routing threshold:** A* edge-weight scoring must complete in <500 ms for all
> Toronto road segments. This model {'**meets**' if batch_total_ms < 500 else '**exceeds**'} that threshold.