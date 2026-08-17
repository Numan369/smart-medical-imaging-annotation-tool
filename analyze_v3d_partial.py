"""Analyse the partial V3D flip-TTA validation CSV (read-only, no modifications)."""
import csv
import statistics

CSV_PATH = "diagnostics_v3d_flip_tta_validation_local/v3d_flip_tta_partial.csv"
THRESHOLD = 0.35
TOTAL_EXPECTED = 1205

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
positives = [r for r in rows if r["is_positive"] == "True"]
negatives  = [r for r in rows if r["is_positive"] == "False"]

print(f"Cases completed: {len(rows)} / {TOTAL_EXPECTED}  ({100*len(rows)/TOTAL_EXPECTED:.1f}%)")
print(f"  Positives: {len(positives)}   Negatives: {len(negatives)}")
print()

# Standard V3C metrics on this subset
std_pos_dice   = [float(r["standard_dice"]) for r in positives]
tta_pos_dice   = [float(r["tta_dice"])      for r in positives]
std_miss_pos   = sum(1 for r in positives if r["standard_empty_prediction"] == "True")
tta_miss_pos   = sum(1 for r in positives if r["tta_empty_prediction"]      == "True")
std_empty_neg  = sum(1 for r in negatives if r["standard_empty_prediction"] == "True")
tta_empty_neg  = sum(1 for r in negatives if r["tta_empty_prediction"]      == "True")

def safe_mean(lst):
    return statistics.mean(lst) if lst else float("nan")

print("=== Standard V3C (partial subset) ===")
print(f"  Positive Dice:          {safe_mean(std_pos_dice):.6f}")
print(f"  Positive misses:        {std_miss_pos}/{len(positives)}  ({100*std_miss_pos/max(1,len(positives)):.1f}%)")
print(f"  Neg empty-mask acc:     {std_empty_neg}/{len(negatives)} = {std_empty_neg/max(1,len(negatives)):.6f}")
print()
print("=== V3D Flip-TTA (partial subset) ===")
print(f"  Positive Dice:          {safe_mean(tta_pos_dice):.6f}")
print(f"  Positive misses:        {tta_miss_pos}/{len(positives)}  ({100*tta_miss_pos/max(1,len(positives)):.1f}%)")
print(f"  Neg empty-mask acc:     {tta_empty_neg}/{len(negatives)} = {tta_empty_neg/max(1,len(negatives)):.6f}")
print()

# Delta statistics
pos_deltas = [float(r["dice_delta_tta_minus_standard"]) for r in positives]
neg_deltas = [float(r["dice_delta_tta_minus_standard"]) for r in negatives]
improved   = sum(1 for d in pos_deltas if d > 0)
worsened   = sum(1 for d in pos_deltas if d < 0)
unchanged  = sum(1 for d in pos_deltas if d == 0)
fp_introduced = sum(
    1 for r in negatives
    if r["standard_empty_prediction"] == "True"
    and r["tta_empty_prediction"] == "False"
)
fp_recovered = sum(
    1 for r in negatives
    if r["standard_empty_prediction"] == "False"
    and r["tta_empty_prediction"] == "True"
)

print("=== Delta: TTA minus Standard ===")
print(f"  Positive Dice mean:     {safe_mean(pos_deltas):+.6f}")
if pos_deltas:
    print(f"  Positive Dice median:   {statistics.median(pos_deltas):+.6f}")
print(f"  Improved positives:     {improved}")
print(f"  Worsened positives:     {worsened}")
print(f"  Unchanged positives:    {unchanged}")
print(f"  FP introduced on negs:  {fp_introduced}  (std=clean, TTA=FP)")
print(f"  FP recovered on negs:   {fp_recovered}  (std=FP, TTA=clean)")
print()

# Max probability for standard complete misses
std_misses = [r for r in positives if r["standard_empty_prediction"] == "True"]
print(f"=== Standard complete misses  n={len(std_misses)} ===")
if std_misses:
    max_probs = [float(r["standard_max_probability"]) for r in std_misses]
    print(f"  Max prob range:         {min(max_probs):.4f} – {max(max_probs):.4f}")
    print(f"  Mean max prob:          {safe_mean(max_probs):.4f}")
    print(f"  Median max prob:        {statistics.median(max_probs):.4f}")
    below = sum(1 for p in max_probs if p < THRESHOLD)
    near  = sum(1 for p in max_probs if 0.25 <= p < THRESHOLD)
    print(f"  Max prob < {THRESHOLD}:       {below}/{len(max_probs)}")
    print(f"  Max prob in [0.25,{THRESHOLD}): {near}/{len(max_probs)}")
    print()
    print("  Per-miss detail (sorted by max_prob):")
    for r in sorted(std_misses, key=lambda x: float(x["standard_max_probability"])):
        print(f"    {r['image_id'][:30]:30s}  max_p={float(r['standard_max_probability']):.4f}"
              f"  bin={r['lesion_size_bin']:10s}  tta_dice={float(r['tta_dice']):.4f}")

print()

# Lesion-size breakdown
bins = ["tiny", "small", "medium", "large"]
print("=== Positive Dice by lesion size ===")
for b in bins:
    group = [r for r in positives if r["lesion_size_bin"] == b]
    if not group:
        continue
    sd = [float(r["standard_dice"]) for r in group]
    td = [float(r["tta_dice"])      for r in group]
    print(f"  {b:8s}  n={len(group):3d}  std={safe_mean(sd):.4f}  tta={safe_mean(td):.4f}"
          f"  delta={safe_mean(td)-safe_mean(sd):+.4f}")

print()
print("NOTE: These results are from a partial run "
      f"({len(rows)}/{TOTAL_EXPECTED} images). "
      "Full results will differ.")
