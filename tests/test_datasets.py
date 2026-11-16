"""
Tests for the external-dataset loaders (training/datasets/).

Each loader maps a dataset onto the canonical columns velocity_features() reads.
These use tiny hand-made frames, so they run in CI without downloading anything.
"""
import pandas as pd

from training.build_features import velocity_features
from training.datasets import CANONICAL_COLUMNS, sparkov


def _sparkov_raw():
    # Deliberately out of time order: the loader must sort.
    return pd.DataFrame({
        "trans_date_trans_time": ["2019-01-01 10:30:00", "2019-01-01 10:00:00",
                                  "2019-01-01 11:00:00"],
        "cc_num": [111, 111, 222],
        "merchant": ["fraud_A", "fraud_A", "fraud_B"],
        "category": ["grocery_pos", "grocery_pos", "travel"],
        "amt": [20.0, 10.0, 500.0],
        "gender": ["F", "F", "M"],
        "dob": ["1990-01-01", "1990-01-01", "1970-06-15"],
        "city_pop": [1000, 1000, 50000],
        "zip": [10001, 10001, 94105],
        "lat": [40.7, 40.7, 37.8], "long": [-74.0, -74.0, -122.4],
        "merch_lat": [40.7, 40.7, 40.7], "merch_long": [-74.0, -74.0, -74.0],
        "is_fraud": [0, 0, 1],
    })


class TestSparkov:
    def test_canonical_columns_sorted_by_time(self):
        canonical, static = sparkov.to_canonical(_sparkov_raw())
        assert list(canonical.columns) == CANONICAL_COLUMNS
        assert canonical["purchase_time"].is_monotonic_increasing
        assert canonical["purchase_value"].tolist() == [10.0, 20.0, 500.0]
        assert len(static) == len(canonical)

    def test_mapping(self):
        canonical, static = sparkov.to_canonical(_sparkov_raw())
        assert canonical["user_id"].tolist() == ["111", "111", "222"]
        assert canonical["device_id"].tolist() == ["fraud_A", "fraud_A", "fraud_B"]
        assert canonical["ip_country"].tolist() == ["grocery_pos", "grocery_pos", "travel"]
        assert static["merchant_distance_km"].iloc[0] == 0
        assert static["merchant_distance_km"].iloc[2] > 3000   # SF card, NY merchant

    def test_velocity_sees_only_the_past(self):
        canonical, _ = sparkov.to_canonical(_sparkov_raw())
        v = velocity_features(canonical)
        # Card 111: first purchase sees no history, second sees the first ($10).
        assert v["purchase_cnt_24h"].tolist() == [0, 1, 0]
        assert v["purchase_amt_24h"].tolist() == [0, 10.0, 0]
        assert v["time_since_last_txn_uid"].tolist() == [-1, 1800, -1]   # -1 = no history
