import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usedcars.clean import (CATEGORICAL, FEATURES, add_features, drop_reposts,  # noqa: E402
                            filter_plausible, normalise_model, prepare, price_bands)


def listing(**overrides):
    row = dict(price=12000, year=2015.0, manufacturer="toyota", model="Camry", condition="good",
               cylinders="4 cylinders", fuel="gas", odometer=80000.0, title_status="clean",
               transmission="automatic", drive="fwd", type="sedan", paint_color="white",
               state="ca", VIN=None, posting_date="2021-04-20")
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_filter_removes_placeholder_and_impossible_listings():
    raw = frame(listing(), listing(price=0), listing(price=3_000_000), listing(year=1950.0),
                listing(year=np.nan), listing(odometer=5.0), listing(odometer=np.nan))
    kept, removed = filter_plausible(raw)
    assert len(kept) == 1
    assert removed == {"price outside range": 2, "year missing or outside range": 2,
                       "odometer missing or outside range": 2}


def test_reposts_with_the_same_vin_are_one_car():
    raw = frame(listing(VIN="A1", state="ca"), listing(VIN="A1", state="nv"), listing(VIN="B2", odometer=90000.0))
    assert len(drop_reposts(raw)) == 2


def test_reposts_without_a_vin_are_matched_on_the_car_details():
    raw = frame(listing(state="ca"), listing(state="or"), listing(price=12500))
    assert len(drop_reposts(raw)) == 2


def test_a_car_listed_with_and_without_its_vin_is_one_car():
    raw = frame(listing(VIN="A1"), listing(VIN=None))
    assert len(drop_reposts(raw)) == 1


def test_model_names_are_normalised_and_rare_ones_grouped():
    models = pd.Series(["Camry", "camry ", "CAMRY", "Civic", "one-off  special", None])
    assert normalise_model(models, top_n=2).tolist() == ["camry", "camry", "camry", "civic", "other", "other"]


def test_add_features_builds_age_and_fills_missing_categories():
    out = add_features(frame(listing(year=2011.0, condition=None, drive=None)))
    assert list(out.columns) == FEATURES + ["price"]
    assert out.loc[0, "age"] == 10
    assert out.loc[0, "condition"] == "unknown" and out.loc[0, "drive"] == "unknown"
    assert all(isinstance(out[c].dtype, pd.CategoricalDtype) for c in CATEGORICAL)


def test_prepare_reports_what_was_removed():
    raw = frame(listing(VIN="A1"), listing(VIN="A1", state="nv"), listing(price=0), listing(odometer=95000.0))
    data, removed = prepare(raw)
    assert len(data) == 2
    assert removed["price outside range"] == 1 and removed["reposts of the same car"] == 1


def test_price_bands():
    assert price_bands(pd.Series([4999, 5000, 19999, 20000, 50000])).tolist() == [
        "under $5k", "$5k-10k", "$10k-20k", "$20k-35k", "over $35k"]
