"""
Benchmark Comparison: Ground Truth vs Algorithm Output
======================================================
Compares real player attributes (ground truth) against the values
produced by the attribute calculation algorithm.

Ground truth provided by user (2026-04-29).
"""

import json
import os

# ─── Ground Truth ────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    "player_5": {
        "pace":        170,
        "acceleration": 150,
        "work_rate":   140,
        "stamina":     120,
    },
    "player_14": {
        "pace":        150,
        "acceleration": 150,
        "work_rate":   140,
        "stamina":     140,
    },
    "player_20": {
        "pace":        120,
        "acceleration": 130,
        "work_rate":   150,
        "stamina":     130,
    },
}

ATTR_ORDER = ["pace", "acceleration", "work_rate", "stamina"]

# ─── Load Algorithm Output ───────────────────────────────────────────────────
MERGED_JSON = os.path.join(
    os.path.dirname(__file__),
    "mergeJson", "merged_players_attributes.json"
)

with open(MERGED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

algorithm_output = {}
for pid, pdata in data["players"].items():
    algorithm_output[pid] = pdata["attributes"]

# ─── Comparison ──────────────────────────────────────────────────────────────
def error_label(diff: int) -> str:
    """Return a qualitative rating based on absolute error."""
    abs_diff = abs(diff)
    if abs_diff <= 5:
        return "EXCELLENT"
    elif abs_diff <= 15:
        return "GOOD"
    elif abs_diff <= 30:
        return "FAIR"
    else:
        return "POOR"


def compare_player(player_id: str, gt: dict, pred: dict) -> dict:
    results = {}
    for attr in ATTR_ORDER:
        gt_val   = gt.get(attr, None)
        pred_val = pred.get(attr, None)
        if gt_val is None or pred_val is None:
            results[attr] = {"gt": gt_val, "pred": pred_val, "diff": None, "pct_error": None, "label": "N/A"}
            continue
        diff = pred_val - gt_val
        pct_error = (diff / gt_val) * 100
        results[attr] = {
            "gt":        gt_val,
            "pred":      pred_val,
            "diff":      diff,
            "pct_error": pct_error,
            "label":     error_label(diff),
        }
    return results


# ─── Print Report ────────────────────────────────────────────────────────────
HEADER  = f"  {'Attribute':<16} {'Ground Truth':>12} {'Algorithm':>10} {'Diff':>8} {'Error%':>8}  {'Rating'}"
DIVIDER = "  " + "-" * 70

print("\n" + "=" * 74)
print("     NEXTSTAR AI  -  Attribute Benchmark Report")
print("=" * 74)

summary_errors = {}

for player_id in ["player_5", "player_14", "player_20"]:
    gt   = GROUND_TRUTH.get(player_id, {})
    pred = algorithm_output.get(player_id, {})
    comparison = compare_player(player_id, gt, pred)

    print(f"\n  {player_id.upper()}")
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)

    player_abs_diffs = []
    for attr, vals in comparison.items():
        if vals["diff"] is not None:
            sign      = "+" if vals["diff"] >= 0 else ""
            diff_str  = f"{sign}{vals['diff']}"
            pct_str   = f"{sign}{vals['pct_error']:.1f}%"
            player_abs_diffs.append(abs(vals["diff"]))
        else:
            diff_str = pct_str = "N/A"

        print(
            f"  {attr:<16} {str(vals['gt']):>12} {str(vals['pred']):>10} "
            f"{diff_str:>8} {pct_str:>8}  {vals['label']}"
        )

    mae  = sum(player_abs_diffs) / len(player_abs_diffs) if player_abs_diffs else 0
    rmse = (sum(d**2 for d in player_abs_diffs) / len(player_abs_diffs)) ** 0.5 if player_abs_diffs else 0
    summary_errors[player_id] = {"mae": mae, "rmse": rmse}
    print(DIVIDER)
    print(f"  MAE = {mae:.1f} pts   |   RMSE = {rmse:.1f} pts   |   Rating: {error_label(int(mae))}")

# ─── Overall Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("  OVERALL SUMMARY")
print("=" * 74)
print(f"\n  {'Player':<12} {'MAE':>8} {'RMSE':>8}   {'Overall Rating'}")
print("  " + "-" * 46)
for pid, errs in summary_errors.items():
    rating = error_label(int(errs["mae"]))
    print(f"  {pid:<12} {errs['mae']:>7.1f}  {errs['rmse']:>7.1f}   {rating}")

all_maes  = [e["mae"]  for e in summary_errors.values()]
all_rmses = [e["rmse"] for e in summary_errors.values()]
overall_mae  = sum(all_maes)  / len(all_maes)
overall_rmse = sum(all_rmses) / len(all_rmses)
print("  " + "-" * 46)
print(f"  {'ALL PLAYERS':<12} {overall_mae:>7.1f}  {overall_rmse:>7.1f}   {error_label(int(overall_mae))}")

# ─── Per-Attribute Analysis ──────────────────────────────────────────────────
print("\n" + "=" * 74)
print("  PER-ATTRIBUTE ANALYSIS (across all 3 players)")
print("=" * 74)
print(f"\n  {'Attribute':<16} {'Avg Err':>9} {'Avg Err%':>10} {'Max Err':>9} {'Max Err%':>10} {'Avg Bias':>10}  {'Rating'}")
print("  " + "-" * 76)

for attr in ATTR_ORDER:
    errors     = []
    pct_errors = []
    biases     = []
    for player_id in ["player_5", "player_14", "player_20"]:
        gt_val   = GROUND_TRUTH[player_id].get(attr)
        pred_val = algorithm_output.get(player_id, {}).get(attr)
        if gt_val is not None and pred_val is not None:
            abs_err = abs(pred_val - gt_val)
            errors.append(abs_err)
            pct_errors.append((abs_err / gt_val) * 100)
            biases.append(pred_val - gt_val)

    if errors:
        avg_e   = sum(errors)     / len(errors)
        avg_pct = sum(pct_errors) / len(pct_errors)
        max_e   = max(errors)
        max_pct = max(pct_errors)
        avg_b   = sum(biases)     / len(biases)
        bias_str    = f"{'+' if avg_b >= 0 else ''}{avg_b:.1f}"
        avg_pct_str = f"{avg_pct:.1f}%"
        max_pct_str = f"{max_pct:.1f}%"
        print(
            f"  {attr:<16} {avg_e:>9.1f} {avg_pct_str:>10} {max_e:>9} {max_pct_str:>10} {bias_str:>10}   {error_label(int(avg_e))}"
        )

print("  " + "-" * 76)
print("\n  Avg Bias: positive = algorithm overestimates, negative = underestimates\n")
print("=" * 74 + "\n")
