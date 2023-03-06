"""dataset utilities"""

from importlib import resources
import pandas as pd


def load_cencus(**kwargs) -> pd.DataFrame:
    with resources.open_text("data", "adult.csv") as f:
        return pd.read_csv(f, **kwargs)
