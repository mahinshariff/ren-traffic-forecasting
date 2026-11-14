"""Command-line entry point: ``python -m renfc <stage> --config <yaml> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, out_dir
from .utils import log, setup_logging, timed

STAGES = ["coverage", "etl", "sweep", "distribution", "commonality", "figures", "all"]


def apply_override(cfg: dict, item: str) -> None:
    """Set ``a.b.c=value`` in the nested config; the value is parsed as YAML."""
    import yaml

    key, _, raw = item.partition("=")
    if not _:
        raise SystemExit(f"--override expects KEY=VALUE, got '{item}'")
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="renfc", description="REN traffic forecasting study pipeline")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", "-c", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--routers", nargs="*", default=None, help="Restrict a stage to these routers")
    p.add_argument("--router-index", type=int, default=None,
                   help="Pick one selected router by index (for SLURM array jobs); overrides --routers")
    p.add_argument("--no-report", action="store_true", help="(sweep) skip the report after the runs")
    p.add_argument("--override", "-o", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config entry, e.g. -o sweep.seeds=[0] -o sweep.train.epochs=20")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    for item in args.override:
        apply_override(cfg, item)
    log_dir = out_dir(cfg, "logs")
    setup_logging(log_dir / f"{args.stage}.log")
    log.info("stage=%s config=%s out_dir=%s", args.stage, cfg["_config_path"], cfg.paths.out_dir)
    for item in args.override:
        log.info("config override: %s", item)

    routers = args.routers
    if args.router_index is not None:
        from .coverage import load_selection

        sel = load_selection(cfg)["routers"]
        routers = [sel[args.router_index]]
        log.info("router index %d -> %s", args.router_index, routers)

    stages = STAGES[:-1] if args.stage == "all" else [args.stage]
    for stage in stages:
        with timed(f"stage {stage}"):
            if stage == "coverage":
                from . import coverage

                coverage.run(cfg, routers)
            elif stage == "etl":
                from . import etl

                etl.run(cfg, routers)
            elif stage == "sweep":
                from . import experiment

                experiment.run(cfg, routers, report=not args.no_report)
            elif stage == "distribution":
                from . import distribution

                distribution.run(cfg, routers)
            elif stage == "commonality":
                from . import commonality

                commonality.run(cfg, routers)
            elif stage == "figures":
                from . import figures

                figures.run(cfg, routers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
