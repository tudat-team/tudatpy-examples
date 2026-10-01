import importlib.util
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

path = Path(__file__).with_name("mpc_radar_gaia_estimation.py")
spec = importlib.util.spec_from_file_location("example", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_cardinal_scan_directions_and_cos_declination():
    residuals = np.array([[2.0, 3.0], [2.0, 3.0]])
    covariance = np.tile(np.diag([4.0, 9.0]), (2, 1, 1))
    projected, sigma = example.project_gaia_residuals(
        residuals, covariance, np.full(2, np.pi / 3), np.array([0.0, np.pi / 2])
    )
    np.testing.assert_allclose(projected, [[3, 1], [1, -3]], atol=1e-14)
    np.testing.assert_allclose(sigma, [[3, 1], [1, 3]], atol=1e-14)


def test_correlated_uncertainty_projection():
    # Equal-coordinate errors with positive correlation: the sum direction
    # has larger variance than the difference direction.
    covariance = np.array([[[4.0, 3.0], [3.0, 4.0]]])
    projected, sigma = example.project_gaia_residuals(
        np.array([[1.0, 1.0]]), covariance, np.zeros(1), np.array([np.pi / 4])
    )
    np.testing.assert_allclose(projected, [[np.sqrt(2), 0]], atol=1e-14)
    np.testing.assert_allclose(sigma, [[np.sqrt(7), 1]], atol=1e-14)


def test_shared_transit_covariance_uses_first_ccd():
    table = pd.DataFrame(
        {
            "transit_id": [1, 1, 2],
            "ra_error_random": [2, 3, 4],
            "dec_error_random": [3, 4, 5],
            "ra_dec_correlation_random": [0, 0, 0],
            "ra_error_systematic": [4, 99, 6],
            "dec_error_systematic": [5, 99, 7],
            "ra_dec_correlation_systematic": [0.5, 0.9, -0.5],
        }
    )
    covariance = example.gaia_marginal_covariance(table)
    np.testing.assert_allclose(
        covariance, [[[20, 10], [10, 34]], [[25, 10], [10, 41]], [[52, -21], [-21, 74]]]
    )


def test_residual_alignment_uses_event_ids_not_scalar_adjacency():
    class Dataset:
        def get_data(self, *args, **kwargs):
            return {
                "times": [20.0, 10.0, 5.0],
                "observation_ids": [12, 11, 90],
                "set_ids": [1, 1, 2],
                "scalar_components": [
                    (12, 1),
                    (90, 0),
                    (11, 0),
                    (12, 0),
                    (90, 1),
                    (11, 1),
                ],
                "metadata": {
                    1: {
                        "link_definition": example.observable_models_setup.links.LinkDefinition(
                            {
                                example.observable_models_setup.links.transmitter: example.observable_models_setup.links.body_origin_link_end_id(
                                    "673"
                                ),
                                example.observable_models_setup.links.receiver: example.observable_models_setup.links.body_origin_link_end_id(
                                    "Gaia"
                                ),
                            }
                        )
                    },
                    2: {
                        "link_definition": example.observable_models_setup.links.LinkDefinition(
                            {
                                example.observable_models_setup.links.transmitter: example.observable_models_setup.links.body_origin_link_end_id(
                                    "673"
                                ),
                                example.observable_models_setup.links.receiver: example.observable_models_setup.links.body_origin_link_end_id(
                                    "Earth"
                                ),
                            }
                        )
                    },
                },
            }

    table = pd.DataFrame(
        {
            "epoch": [10.0, 20.0],
            "dec": [0.0, 0.0],
            "position_angle_scan": [0.0, np.pi / 2],
            "transit_id": [1, 2],
            "ra_error_random": [1.0, 1.0],
            "dec_error_random": [1.0, 1.0],
            "ra_dec_correlation_random": [0.0, 0.0],
            "ra_error_systematic": [1.0, 1.0],
            "dec_error_systematic": [1.0, 1.0],
            "ra_dec_correlation_systematic": [0.0, 0.0],
        }
    )
    result = example.gaia_residual_data(
        SimpleNamespace(final_residuals=[4, 999, 1, 3, 999, 2]),
        Dataset(),
        SimpleNamespace(table=table),
    )
    np.testing.assert_array_equal(result["scalar_indices"], [[2, 5], [3, 0]])
    np.testing.assert_allclose(result["residuals"], [[1, 2], [3, 4]])
    np.testing.assert_allclose(result["scan_residuals"], [[2, 1], [3, -4]], atol=1e-14)


def test_missing_gaia_ccds_are_rejected():
    with pytest.raises(RuntimeError, match="nonempty"):
        example.gaia_residual_data(
            None, None, SimpleNamespace(table=pd.DataFrame({"epoch": []}))
        )


def test_bennu_cannot_silently_run_without_gaia(monkeypatch):
    def missing_data(*args, **kwargs):
        raise RuntimeError("No observations found for [101955]")

    monkeypatch.setattr(example, "TARGET", "101955")
    monkeypatch.setattr(example, "GAIA_ARCHIVE_PATH", None)
    monkeypatch.setattr(example.GaiaAstrometry, "load_from_astroquery", missing_data)
    with pytest.raises(RuntimeError, match="No observations found"):
        example.load_gaia_astrometry()
