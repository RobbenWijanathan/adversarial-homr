#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EDA graphs comparing the surrogate before vs after PGD defense."
    )
    parser.add_argument("--comparison-dir", type=Path, default=Path("results/surrogate_comparison_a100"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--notebook", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def autoattack_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    records = []
    for entry in payload["results"]:
        before = entry["before_defense"]
        after = entry["after_defense"]
        records.append(
            {
                "epsilon": entry["epsilon"],
                "before_count_accuracy": before["robust_count_accuracy"],
                "after_count_accuracy": after["robust_count_accuracy"],
                "before_token_accuracy": before["robust_token_accuracy"],
                "after_token_accuracy": after["robust_token_accuracy"],
                "before_mean_ser": before["robust_mean_ser"],
                "after_mean_ser": after["robust_mean_ser"],
            }
        )
    return pd.DataFrame(records).sort_values("epsilon").reset_index(drop=True)


def pgd_grid_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    records = []
    for entry in payload["results"]:
        records.append(
            {
                "epsilon": entry["epsilon"],
                "before_overall_accuracy": entry["clean"]["overall_accuracy"],
                "after_overall_accuracy": entry["pgd"]["overall_accuracy"],
                "before_mean_ser": entry["clean"]["mean_ser"],
                "after_mean_ser": entry["pgd"]["mean_ser"],
            }
        )
    return pd.DataFrame(records).sort_values("epsilon").reset_index(drop=True)


def per_branch_at_max_epsilon(payload: dict[str, Any]) -> pd.DataFrame:
    entry = max(payload["results"], key=lambda row: row["epsilon"])
    branches = sorted(entry["clean"]["branch_accuracy"].keys())
    records = []
    for branch in branches:
        records.append(
            {
                "branch": branch,
                "before_accuracy": entry["clean"]["branch_accuracy"][branch],
                "after_accuracy": entry["pgd"]["branch_accuracy"][branch],
                "before_ser": entry["clean"]["branch_ser"][branch],
                "after_ser": entry["pgd"]["branch_ser"][branch],
            }
        )
    return pd.DataFrame(records), entry["epsilon"]


def line_plot(frame: pd.DataFrame, before_column: str, after_column: str, title: str, ylabel: str, out_path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(frame["epsilon"], frame[before_column], marker="o", label="before defense (clean surrogate)")
    axes.plot(frame["epsilon"], frame[after_column], marker="s", label="after defense (PGD surrogate)")
    axes.set_xlabel("L-inf epsilon")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def grouped_bar_plot(frame: pd.DataFrame, before_column: str, after_column: str, title: str, ylabel: str, out_path: Path) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    positions = range(len(frame))
    width = 0.38
    axes.bar([p - width / 2 for p in positions], frame[before_column], width, label="before defense")
    axes.bar([p + width / 2 for p in positions], frame[after_column], width, label="after defense")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(frame["branch"], rotation=20)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, axis="y", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def write_markdown_summary(out_dir: Path, autoattack: pd.DataFrame | None, pgd_grid: pd.DataFrame | None) -> Path:
    lines = ["# Surrogate robustness: before vs after PGD defense", ""]
    if autoattack is not None:
        lines.append("## AutoAttack (white-box L-inf) on the staff-count head")
        lines.append("")
        lines.append(autoattack.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        clean_drop = autoattack["before_count_accuracy"].iloc[0] - autoattack["before_count_accuracy"].iloc[-1]
        pgd_drop = autoattack["after_count_accuracy"].iloc[0] - autoattack["after_count_accuracy"].iloc[-1]
        lines.append(
            f"Staff-count accuracy drop from epsilon {autoattack['epsilon'].iloc[0]:.3f} to "
            f"{autoattack['epsilon'].iloc[-1]:.3f}: before defense {clean_drop:.4f}, after defense {pgd_drop:.4f}."
        )
        lines.append("")
    if pgd_grid is not None:
        lines.append("## PGD epsilon-grid (token-level)")
        lines.append("")
        lines.append(pgd_grid.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    summary_path = out_dir / "eda_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_notebook(out_dir: Path, figure_paths: list[Path], summary_path: Path, notebook_path: Path) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    cells = [
        new_markdown_cell(
            "# Adversarial HOMR surrogate: before vs after PGD defense\n\n"
            "Generated EDA comparing the clean surrogate (before defense) and the PGD adversarially "
            "trained surrogate (after defense) under AutoAttack and a PGD epsilon grid."
        ),
        new_markdown_cell(summary_path.read_text(encoding="utf-8")),
    ]
    for figure_path in figure_paths:
        cells.append(new_markdown_cell(f"## {figure_path.stem.replace('_', ' ')}"))
        cells.append(
            new_code_cell(
                "from IPython.display import Image\n" f"Image(filename={str(figure_path.name)!r})"
            )
        )
    notebook = new_notebook(cells=cells)
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.comparison_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    autoattack_payload = load_json(args.comparison_dir / "autoattack_comparison.json")
    pgd_payload = load_json(args.comparison_dir / "comparison.json")

    figure_paths: list[Path] = []
    autoattack_frame = None
    pgd_frame = None

    if autoattack_payload is not None:
        autoattack_frame = autoattack_dataframe(autoattack_payload)
        autoattack_frame.to_csv(out_dir / "autoattack_metrics.csv", index=False)

        accuracy_path = out_dir / "autoattack_count_accuracy_vs_epsilon.png"
        line_plot(
            autoattack_frame,
            "before_count_accuracy",
            "after_count_accuracy",
            "AutoAttack: staff-count accuracy under attack",
            "robust staff-count accuracy",
            accuracy_path,
        )
        figure_paths.append(accuracy_path)

        token_path = out_dir / "autoattack_token_accuracy_vs_epsilon.png"
        line_plot(
            autoattack_frame,
            "before_token_accuracy",
            "after_token_accuracy",
            "AutoAttack: token accuracy under attack",
            "robust token accuracy",
            token_path,
        )
        figure_paths.append(token_path)

        ser_path = out_dir / "autoattack_ser_vs_epsilon.png"
        line_plot(
            autoattack_frame,
            "before_mean_ser",
            "after_mean_ser",
            "AutoAttack: symbol error rate under attack",
            "robust mean SER",
            ser_path,
        )
        figure_paths.append(ser_path)

    if pgd_payload is not None:
        pgd_frame = pgd_grid_dataframe(pgd_payload)
        pgd_frame.to_csv(out_dir / "pgd_grid_metrics.csv", index=False)

        pgd_accuracy_path = out_dir / "pgd_grid_token_accuracy_vs_epsilon.png"
        line_plot(
            pgd_frame,
            "before_overall_accuracy",
            "after_overall_accuracy",
            "PGD grid: overall token accuracy under attack",
            "overall token accuracy",
            pgd_accuracy_path,
        )
        figure_paths.append(pgd_accuracy_path)

        pgd_ser_path = out_dir / "pgd_grid_ser_vs_epsilon.png"
        line_plot(
            pgd_frame,
            "before_mean_ser",
            "after_mean_ser",
            "PGD grid: symbol error rate under attack",
            "mean SER",
            pgd_ser_path,
        )
        figure_paths.append(pgd_ser_path)

        branch_frame, max_epsilon = per_branch_at_max_epsilon(pgd_payload)
        branch_frame.to_csv(out_dir / "per_branch_at_max_epsilon.csv", index=False)
        branch_path = out_dir / "per_branch_accuracy_at_max_epsilon.png"
        grouped_bar_plot(
            branch_frame,
            "before_accuracy",
            "after_accuracy",
            f"Per-branch accuracy at epsilon {max_epsilon:.3f}",
            "branch accuracy",
            branch_path,
        )
        figure_paths.append(branch_path)

    summary_path = write_markdown_summary(out_dir, autoattack_frame, pgd_frame)

    notebook_path = args.notebook or (out_dir / "eda_comparison.ipynb")
    build_notebook(out_dir, figure_paths, summary_path, notebook_path)

    print(f"wrote {len(figure_paths)} figures to {out_dir}")
    for figure_path in figure_paths:
        print(f"  {figure_path}")
    print(f"summary: {summary_path}")
    print(f"notebook: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
