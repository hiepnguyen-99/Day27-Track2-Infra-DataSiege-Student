"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)
    ctx.state["history"] = {
        "data_batch": [],
        "embedding_batch": [],
    }


def _ok(result):
    return isinstance(result, dict) and "error" not in result


def _append_history(ctx, key, item):
    ctx.state.setdefault("history", {}).setdefault(key, []).append(item)


def _rolling_zscores(history, values):
    if len(history) < 5:
        return [0.0 for _ in values]
    means = []
    stds = []
    for idx in range(len(values)):
        series = [item[idx] for item in history]
        mean = sum(series) / len(series)
        variance = sum((value - mean) * (value - mean) for value in series) / len(series)
        means.append(mean)
        stds.append(variance ** 0.5)
    out = []
    for idx, value in enumerate(values):
        std = stds[idx] if stds[idx] > 1e-9 else 1e-9
        out.append(abs(value - means[idx]) / std)
    return out


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if not _ok(profile):
        return Verdict(alert=False, pillar="checks", reason="batch_profile unavailable")

    b = ctx.baseline
    row_count = profile["row_count"]
    null_rate = profile["null_rate"].get("customer_id", 0.0)
    mean_amount = profile["mean_amount"]
    std_amount = profile["std_amount"]
    staleness = profile["staleness_min"]

    reasons = []
    if row_count < b["row_count_min"] or row_count > b["row_count_max"]:
        reasons.append("row_count_out_of_range")
    if null_rate > b["null_rate_max"]:
        reasons.append("null_rate_high")
    if mean_amount < b["mean_amount_min"] or mean_amount > b["mean_amount_max"]:
        reasons.append("mean_amount_out_of_range")
    if staleness > b["staleness_min_max"]:
        reasons.append("staleness_high")

    history = ctx.state["history"]["data_batch"]
    values = (row_count, null_rate, mean_amount, std_amount, staleness)
    zscores = _rolling_zscores(history, values)

    high_edge_combo = (
        row_count > 0.985 * b["row_count_max"]
        and mean_amount > 0.98 * b["mean_amount_max"]
    )
    low_edge_combo = (
        row_count < 1.015 * b["row_count_min"]
        and mean_amount < 1.02 * b["mean_amount_min"]
    )
    multi_signal_shift = sum(score > 1.8 for score in (zscores[0], zscores[2], zscores[4])) >= 2

    _append_history(ctx, "data_batch", values)

    if reasons or high_edge_combo or low_edge_combo or multi_signal_shift:
        if high_edge_combo:
            reasons.append("near_limit_distribution_shift")
        if low_edge_combo:
            reasons.append("near_limit_low_distribution_shift")
        if multi_signal_shift:
            reasons.append("rolling_distribution_anomaly")
        return Verdict(alert=True, pillar="checks", confidence=0.92, reason=";".join(reasons))
    return Verdict(alert=False, pillar="checks", confidence=0.15, reason="within_batch_guardrails")


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if not _ok(diff):
        return Verdict(alert=False, pillar="contracts", reason="contract_diff unavailable")

    reasons = list(diff.get("violations", []))
    if diff["freshness_delay_min"] > ctx.baseline["freshness_delay_max_min"]:
        reasons.append("freshness_delay_high")
    if reasons:
        return Verdict(alert=True, pillar="contracts", confidence=0.98, reason=";".join(reasons))
    return Verdict(alert=False, pillar="contracts", confidence=0.1, reason="contract_clean")


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if not _ok(graph):
        return Verdict(alert=False, pillar="lineage", reason="lineage_graph_slice unavailable")

    reasons = []
    if graph["duration_ms"] > ctx.baseline["lineage_duration_ms_max"]:
        reasons.append("duration_high")
    if graph["actual_upstream"] != ["raw.orders", "raw.customers"]:
        reasons.append("unexpected_upstream")
    if graph["actual_downstream_count"] != 1:
        reasons.append("unexpected_downstream_count")
    if reasons:
        return Verdict(alert=True, pillar="lineage", confidence=0.97, reason=";".join(reasons))
    return Verdict(alert=False, pillar="lineage", confidence=0.1, reason="lineage_matches_expected")


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if not _ok(drift):
        return Verdict(alert=False, pillar="ai_infra", reason="feature_drift unavailable")

    shift = drift["mean_shift_sigma"]
    if shift > 1.0:
        return Verdict(alert=True, pillar="ai_infra", confidence=0.95, reason="feature_mean_shift_high")
    return Verdict(alert=False, pillar="ai_infra", confidence=0.12, reason="feature_drift_normal")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if not _ok(drift):
        return Verdict(alert=False, pillar="ai_infra", reason="embedding_drift unavailable")

    b = ctx.baseline
    shift = drift["centroid_shift"]
    age = drift["avg_doc_age_days"]
    reasons = []
    if shift > b["embedding_centroid_shift_max"]:
        reasons.append("centroid_shift_high")
    if age > b["corpus_avg_doc_age_days_max"]:
        reasons.append("doc_age_high")
    if shift >= 0.039:
        reasons.append("centroid_shift_near_limit")
    if age >= 45.0:
        reasons.append("doc_age_near_limit")

    history = ctx.state["history"]["embedding_batch"]
    values = (shift, age)
    _append_history(ctx, "embedding_batch", values)

    if reasons:
        return Verdict(alert=True, pillar="ai_infra", confidence=0.93, reason=";".join(reasons))
    return Verdict(alert=False, pillar="ai_infra", confidence=0.15, reason="embedding_drift_normal")
