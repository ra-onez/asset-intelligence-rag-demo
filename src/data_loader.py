from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data():
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    work_orders = pd.read_csv(DATA_DIR / "work_orders.csv")
    inspections = pd.read_csv(DATA_DIR / "inspections.csv")
    failures = pd.read_csv(DATA_DIR / "failure_events.csv")

    work_orders["planned_date"] = pd.to_datetime(work_orders["planned_date"])
    work_orders["completed_date"] = pd.to_datetime(
        work_orders["completed_date"], errors="coerce"
    )
    inspections["inspection_date"] = pd.to_datetime(inspections["inspection_date"])
    failures["failure_date"] = pd.to_datetime(failures["failure_date"])

    return assets, work_orders, inspections, failures
