from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from openpyxl import Workbook


def _safe_sheet_name(name: str) -> str:
    # Excel limits: 31 chars, no []:*?/\
    bad = set('[]:*?/\\')
    cleaned = "".join("_" if c in bad else c for c in name)
    cleaned = cleaned.strip() or "Sheet"
    return cleaned[:31]


def csv_dir_to_xlsx(csv_dir: Path, out_xlsx: Path, *, csv_glob: str = "*.csv") -> Path:
    csv_dir = csv_dir.expanduser().resolve()
    out_xlsx = out_xlsx.expanduser().resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(csv_dir.glob(csv_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir} matching {csv_glob}")

    wb = Workbook()
    # remove default sheet
    wb.remove(wb.active)

    for p in csv_paths:
        sheet = wb.create_sheet(title=_safe_sheet_name(p.stem))
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                sheet.append(row)

    wb.save(out_xlsx)
    return out_xlsx


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Convert all CSV files in a directory into a single .xlsx workbook.")
    ap.add_argument("--csv-dir", default="data/outputs", help="Directory containing CSV files")
    ap.add_argument("--out", default="data/outputs/outputs.xlsx", help="Output .xlsx path")
    args = ap.parse_args()

    out = csv_dir_to_xlsx(Path(args.csv_dir), Path(args.out))
    print(str(out))


if __name__ == "__main__":
    main()

