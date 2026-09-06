import pandas as pd

from mircid.tfa.benchmark import evaluate


def test_known_tf_is_ranked_first() -> None:
    activities = pd.DataFrame({"TF1": [4.0], "TF2": [0.1]}, index=["s1"])
    metadata = pd.DataFrame({"sample_id": ["s1"], "perturbed_tf": ["TF1"], "effect": ["activation"]})
    result = evaluate(activities, metadata, top_fraction=0.3)
    assert result.loc[0, "rank"] == 1
    assert bool(result.loc[0, "success"])

