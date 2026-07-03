# Reflection (<=1 page)

The hardest faults were the subtle `checks` and `ai_infra` cases that stayed just inside the published baseline bounds. Straight thresholding is enough for obvious row-count spikes, freshness delays, contract violations, lineage breaks, and large feature or embedding drift, but it misses cases where several metrics all move toward the edge together without one metric clearly crossing the limit.

To handle that, I used the available budget to inspect every event with its relevant metered tool, then layered conservative near-edge rules and rolling-history checks in `ctx.state`. That approach keeps the detector general: baseline violations still drive most alerts, but the defense can also catch combinations like row count and mean amount both clustering near their upper bound, or document age drifting unusually high relative to earlier embedding batches.

If I had another pass, I would spend more time calibrating the history-based logic against clean variance rather than hand-setting a few near-edge thresholds. The current version is intentionally conservative because false positives are penalized heavily. A stronger next iteration would use a more explicit online anomaly score per event type, with a warm-up period and better separation between stable and transitional behavior.
