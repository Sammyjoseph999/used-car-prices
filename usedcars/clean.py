"""Turn raw Craigslist vehicle listings into a table that can be modelled."""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
               "title_status", "transmission", "drive", "type", "paint_color", "state", "VIN", "posting_date"]

LISTING_YEAR = 2021          # every listing in the file was posted in April or May 2021
PRICE_RANGE = (500, 100_000)
YEAR_RANGE = (1990, 2021)
ODOMETER_RANGE = (100, 400_000)

CATEGORICAL = ["manufacturer", "model", "condition", "cylinders", "fuel", "title_status",
               "transmission", "drive", "type", "paint_color", "state"]
NUMERIC = ["age", "odometer"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "price"


def filter_plausible(df):
    """Drop listings whose price, year or mileage cannot be a real car for sale.

    Zero and one-dollar prices are placeholders ("call for price"), and the
    largest values are typing errors or jokes. Returns the kept rows and a
    count of what each rule removed, applied in order.
    """
    steps = [
        ("price outside range", df["price"].between(*PRICE_RANGE)),
        ("year missing or outside range", df["year"].between(*YEAR_RANGE)),
        ("odometer missing or outside range", df["odometer"].between(*ODOMETER_RANGE)),
    ]
    removed, keep = {}, pd.Series(True, index=df.index)
    for name, ok in steps:
        removed[name] = int((keep & ~ok).sum())
        keep &= ok
    return df[keep].copy(), removed


def drop_reposts(df):
    """Keep one row per car.

    Dealers post the same car in several regions. Left in, the same car lands
    in both the training and the test set and the test score looks better than
    it is. A car is identified by its VIN when there is one, and otherwise by
    make, model, year, mileage and price.
    """
    has_vin = df["VIN"].notna()
    with_vin = df[has_vin].drop_duplicates(subset="VIN")
    without_vin = df[~has_vin].drop_duplicates(subset=["manufacturer", "model", "year", "odometer", "price"])
    # A car listed once with a VIN and once without still counts as one car
    key = ["manufacturer", "model", "year", "odometer", "price"]
    return pd.concat([with_vin, without_vin]).drop_duplicates(subset=key).sort_index()


def normalise_model(series, top_n=250):
    """Lower-case the free-text model name and keep only the most common ones.

    The model field is typed by hand and has tens of thousands of spellings.
    Everything outside the top_n becomes "other".
    """
    cleaned = series.fillna("unknown").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    common = cleaned.value_counts().head(top_n).index
    return cleaned.where(cleaned.isin(common), "other")


def add_features(df, top_models=250):
    """Add age, tidy the categorical columns and return only the modelling columns."""
    out = df.copy()
    out["age"] = LISTING_YEAR - out["year"]
    out["model"] = normalise_model(out["model"], top_models)
    for column in CATEGORICAL:
        if column != "model":
            out[column] = out[column].fillna("unknown").str.lower()
        out[column] = out[column].astype("category")
    return out[FEATURES + [TARGET]].reset_index(drop=True)


def prepare(raw, top_models=250):
    """Full pipeline: plausible listings, one row per car, model-ready columns."""
    plausible, removed = filter_plausible(raw)
    unique = drop_reposts(plausible)
    removed["reposts of the same car"] = len(plausible) - len(unique)
    return add_features(unique, top_models), removed


def price_bands(prices):
    """Label prices with the band used in the error analysis."""
    bins = [0, 5_000, 10_000, 20_000, 35_000, np.inf]
    labels = ["under $5k", "$5k-10k", "$10k-20k", "$20k-35k", "over $35k"]
    return pd.cut(prices, bins=bins, labels=labels, right=False)
