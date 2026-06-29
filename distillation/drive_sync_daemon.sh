REPO=/content/adversarial-homr
cd "$REPO"
BATCH_ID="${1:-batch_a100_300}"
INTERVAL="${2:-180}"
B="distillation/batches/$BATCH_ID"
SESS="/content/drive/MyDrive/College/Projects/AI-AOL/adversarial_homr_session"
DEST="$SESS/checkpoint_a100/$BATCH_ID"

mkdir -p "$DEST/teacher_outputs" "$DEST/runs" "$DEST/results"

while true; do
  cp -ru "$B/teacher_outputs/." "$DEST/teacher_outputs/" 2>/dev/null || true
  cp -u "$B/training_manifest.jsonl" "$B/vocab.json" "$DEST/" 2>/dev/null || true
  cp -u "$B"/training_manifest.*.encoded.jsonl "$DEST/" 2>/dev/null || true
  cp -ru distillation/runs/clean_surrogate_a100 distillation/runs/pgd_surrogate_a100 "$DEST/runs/" 2>/dev/null || true
  cp -u results/surrogate_comparison_a100/*.json "$DEST/results/" 2>/dev/null || true
  date -u +"sync %Y-%m-%dT%H:%M:%SZ" >> "$CLAUDE_JOB_DIR/tmp/drive_sync.log"
  sleep "$INTERVAL"
done
