set -e

REPO=/content/adversarial-homr
cd "$REPO"
export PYTHONPATH="$REPO"
export QT_QPA_PLATFORM=offscreen

BATCH_ID="${1:-batch_a100_300}"
EPOCHS="${2:-40}"
B="distillation/batches/$BATCH_ID"

echo "=== [4b] flatten (skip unreadable) ==="
python distillation/flatten_renders.py --images-dir "$B/rendered_images"

echo "=== [5] ONNX teacher on GPU ==="
python distillation/run_onnx_teacher_batch.py \
  --render-log "$B/logs/render_log.jsonl" \
  --teacher-dir "$B/teacher_outputs" --teacher-log "$B/logs/teacher_log.jsonl" \
  --segnet-onnx models/onnx/segnet.onnx --model-dir models/onnx --continue-on-error

echo "=== [6] manifest + splits + vocab ==="
python distillation/build_training_manifest.py --teacher-dir "$B/teacher_outputs" \
  --out "$B/training_manifest.jsonl" --allow-empty
python distillation/make_splits.py --manifest "$B/training_manifest.jsonl" \
  --out-dir "$B/splits" --allow-missing-score-id
python distillation/vocab.py \
  --train-manifest "$B/splits/training_manifest.train.jsonl" \
  --validate-manifest "$B/splits/training_manifest.val.jsonl" \
  --out "$B/vocab.json" --encoded-out-dir "$B"

echo "=== [7] Train ORIGINAL surrogate (before defense) for $EPOCHS epochs ==="
python distillation/train_student.py \
  --train-manifest "$B/training_manifest.train.encoded.jsonl" \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --out-dir distillation/runs/clean_surrogate_a100 \
  --epochs "$EPOCHS" --batch-size 4 --device cuda --quiet --resume

echo "=== [8] Train PGD-DEFENDED surrogate (after defense) for $EPOCHS epochs ==="
python distillation/train_student.py \
  --train-manifest "$B/training_manifest.train.encoded.jsonl" \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --out-dir distillation/runs/pgd_surrogate_a100 \
  --epochs "$EPOCHS" --batch-size 4 --device cuda --quiet --resume \
  --adv-train --pgd-epsilon 0.02 --pgd-steps 10 --pgd-alpha 0.005

echo "=== [9] PGD epsilon-grid comparison ==="
python distillation/evaluate_surrogate.py \
  --clean-checkpoint distillation/runs/clean_surrogate_a100/best_clean.pt \
  --pgd-checkpoint distillation/runs/pgd_surrogate_a100/best_clean.pt \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --epsilon-grid 0.0 0.01 0.02 0.05 0.10 \
  --out-dir results/surrogate_comparison_a100 --device cuda

echo "=== [10] AutoAttack before vs after defense ==="
python distillation/autoattack_eval.py \
  --clean-checkpoint distillation/runs/clean_surrogate_a100/best_clean.pt \
  --pgd-checkpoint distillation/runs/pgd_surrogate_a100/best_clean.pt \
  --val-manifest "$B/training_manifest.val.encoded.jsonl" \
  --epsilon-grid 0.0 0.01 0.02 0.05 0.10 \
  --out-dir results/surrogate_comparison_a100 --device cuda --version standard

echo "=== [10b] EDA graphs ==="
python distillation/eda_comparison.py --comparison-dir results/surrogate_comparison_a100

echo "=== [11] Back up to Drive ==="
SESS="/content/drive/MyDrive/College/Projects/AI-AOL/adversarial_homr_session"
mkdir -p "$SESS/runs" "$SESS/results_a100"
cp -rf distillation/runs/clean_surrogate_a100 distillation/runs/pgd_surrogate_a100 "$SESS/runs/" 2>/dev/null || true
cp -rf results/surrogate_comparison_a100 "$SESS/results_a100/" 2>/dev/null || true
echo "PIPELINE_DONE"
