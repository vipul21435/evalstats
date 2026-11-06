from pathlib import Path

import evalstats


def test_public_names_are_exported():
    for name in evalstats.__all__:
        assert hasattr(evalstats, name), name
    iv = evalstats.mean_interval([1, 0, 1, 1])
    assert iv.estimate == 0.75
    assert evalstats.holm([0.01, 0.04]).tolist() == [0.02, 0.04]


def test_package_ships_type_marker():
    assert (Path(evalstats.__file__).parent / "py.typed").exists()
