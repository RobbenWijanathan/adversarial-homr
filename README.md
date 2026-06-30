# adversarial-homr

Adversarial robustness audit of [HOMR](https://github.com/liebharc/homr), a hierarchical Optical Music Recognition pipeline that converts sheet music images into MusicXML.

## What this does

HOMR chains a convolutional segmentation network, a geometric layout stage, and a transformer decoder. This project probes that pipeline under two perturbation families and builds a differentiable surrogate to study gradient-based attacks and defenses.

**Three tracks:**

- Track A - Spectral noise: frequency-domain 1/f-alpha noise injected at the page level to simulate natural image degradation (scanning artifacts, lighting variation).
- Track B - Square Attack: query-based black-box L-inf attack on prepared staff images with a fixed query budget of 10 per staff.
- Track C - Surrogate: a differentiable end-to-end student distilled from the HOMR ONNX pipeline, hardened with PGD adversarial training, and evaluated with AutoAttack.

**Metrics:** Symbol Error Rate (SER) and Character Error Rate (CER) via Levenshtein distance, measured as drift from the clean pipeline output.

## Key results

HOMR is largely stable under spectral noise (worst-case capped delta SER 0.096). A 10-query Square Attack at epsilon = 0.4 succeeded on 99.7% of staves with capped delta SER 0.635, roughly 6x worse than the worst spectral condition. On the surrogate, PGD adversarial training recovered most token-level degradation and raised staff-count accuracy from 0% to 29% at epsilon = 0.02 under AutoAttack.

## Setup

```bash
poetry install
export PYTHONPATH=$(pwd)
```

ONNX models go in `models/onnx/` (segnet.onnx, tromr_encoder.onnx, tromr_decoder.onnx). Verify with:

```bash
python attacks/src/homr_wrapper.py --describe-only
```

Dataset: [PDMX](https://zenodo.org/records/15571083) - extract to `dataset/mxl/` and `dataset/PDMX.csv`.

## Running experiments

```bash
python attacks/run_spectral_sweep.py   # Track A
python attacks/run_square_sweep.py     # Track B
```

For the surrogate pipeline (Track C), run in order:

```bash
python distillation/build_source_pool.py
python distillation/select_batch.py
python distillation/render_batch.py
python distillation/run_onnx_teacher_batch.py
python distillation/build_training_manifest.py
python distillation/make_splits.py
python distillation/vocab.py
python distillation/train_student.py --device cuda
python distillation/evaluate_surrogate.py
python distillation/autoattack_eval.py
```

## References

- [HOMR](https://github.com/liebharc/homr)
- [Polyphonic-TrOMR](https://github.com/particular-reality/TrOMR)
- [PDMX dataset](https://zenodo.org/records/15571083)
- Andriushchenko et al., Square Attack, ECCV 2020
- Croce & Hein, AutoAttack, ICML 2020
- Madry et al., PGD adversarial training, ICLR 2018
