#!/usr/bin/env python3
"""
CLI-обёртка над DatasetBuilder (Phase 5, раздел 22, 31 — "весь путь можно
воспроизвести одной последовательностью команд").

    python3 scripts/build_dataset.py [--no-persist]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_settings
from src.datasets.builder import build_dataset
from src.db.engine import make_engine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true", help="не писать в match_features, только вывести датасет")
    args = parser.parse_args(argv)

    settings = load_settings()
    engine = make_engine(settings)

    result = build_dataset(engine, persist=not args.no_persist)

    print(f"feature_set_version: {result.feature_set_version}")
    print(f"строк: {len(result.rows)}")
    if not result.dataframe.empty:
        print(result.dataframe.to_string())
    else:
        print("Датасет пуст — проверьте, что ingestion загрузил про-матчи (см. reports/data-quality-report.md).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
