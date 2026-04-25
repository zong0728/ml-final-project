"""Run neural grid for ONE architecture only (used by sbatch array).

    python -m scripts.run_neural_grid_arch --arch gru
"""
import argparse

from scripts.run_neural_grid import (
    register_grid, HPARAM_GRID, ARCHS, _short_name,
)
from src.runner import run_all
from src.training import summarize_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = p.parse_args()

    register_grid()  # registers ALL, but we only run the chosen arch's names.
    names = [_short_name(args.arch, hp) for hp in HPARAM_GRID]
    print(f"[arch={args.arch}] running {len(names)} hparam configs, seeds={args.seeds}")
    for n in names:
        print(f"  - {n}")

    run_all(model_names=names, seeds=args.seeds, skip_on_error=True)

    print(f"\n========= Top configs for arch={args.arch} =========")
    s = summarize_runs()
    s = s[s["model"].str.startswith(f"{args.arch}__")]
    print(s.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
