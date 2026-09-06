def test_embedding_benchmark_imports() -> None:
    from mircid.pathway import run_embedding_benchmark

    assert callable(run_embedding_benchmark.main)


def test_public_predictor_imports() -> None:
    from mircid.hubmir import predict

    assert callable(predict.main)
