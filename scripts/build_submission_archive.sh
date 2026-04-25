#!/usr/bin/env bash
# Build the final submission archive matching the deliverables in the project
# description (see Course Project Description.pdf §4):
#
#   - Code zip          (Canvas)
#   - report.pdf        (Canvas)
#   - submission_24h.csv (Google Form)
#   - submission_48h.csv (Google Form)
#   - policy_selection.txt (Google Form)
#
# Usage (run after finalize_all.sh):
#   bash scripts/build_submission_archive.sh
#
# Outputs all into:  submission_bundle/
set -euo pipefail

OUT=submission_bundle
mkdir -p "$OUT"

# 1. Predictions — copy + rename to clean names.
cp results/submissions/ensemble_pred_24h.csv "$OUT/submission_24h.csv"
cp results/submissions/ensemble_pred_48h.csv "$OUT/submission_48h.csv"

# 2. Policy selection.
cp results/policy_selection.txt "$OUT/policy_selection.txt"

# 3. Report PDF.
if [ -f report/report.pdf ]; then
  cp report/report.pdf "$OUT/report.pdf"
else
  echo "WARN: report/report.pdf not built — recompile and re-run." >&2
fi

# 4. Code zip.
ZIP="$OUT/ml-final-project-code.zip"
rm -f "$ZIP"
git archive --format=zip --output="$ZIP" HEAD
echo "  wrote $ZIP ($(du -h "$ZIP" | cut -f1))"

echo
echo "==== Submission bundle ===="
ls -la "$OUT/"
echo
echo "Sanity checks:"
echo "  24h rows: $(wc -l < "$OUT/submission_24h.csv")  (expected 1993 incl header)"
echo "  48h rows: $(wc -l < "$OUT/submission_48h.csv")  (expected 3986 incl header)"
echo "  policy:   $(cat "$OUT/policy_selection.txt")"
