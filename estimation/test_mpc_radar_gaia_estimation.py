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
        SimpleNamespace(
            best_iteration=0,
            final_residuals=np.zeros(6),
            residual_history=np.column_stack((np.zeros(6), [4, 999, 1, 3, 999, 2])),
            active_flags_per_iteration=np.ones((6, 2), dtype=bool),
        ),
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


def test_missing_gaia_allows_non_gaia_runs(monkeypatch):
    def missing_data(*args, **kwargs):
        raise RuntimeError("No observations found for [101955]")

    monkeypatch.setattr(example, "TARGET", "101955")
    monkeypatch.setattr(example, "GAIA_ARCHIVE_PATH", None)
    monkeypatch.setattr(example.GaiaAstrometry, "load_from_astroquery", missing_data)
    assert example.load_gaia_astrometry() is None


@pytest.mark.parametrize("filter_error", [None, "No observations left after applying filters"])
def test_empty_filtered_gaia_is_allowed(monkeypatch, filter_error):
    def apply_filters(**kwargs):
        if filter_error:
            raise RuntimeError(filter_error)

    gaia = SimpleNamespace(table=pd.DataFrame(), apply_filters=apply_filters)
    monkeypatch.setattr(example.GaiaAstrometry, "load_from_local_archive", lambda *args: gaia)
    assert example.load_gaia_astrometry(archive_path="unused.parquet") is None


def test_gaia_retrieval_failure_is_not_treated_as_empty(monkeypatch):
    def failed_query(*args):
        raise RuntimeError("Archive query failed")

    monkeypatch.setattr(example.GaiaAstrometry, "load_from_local_archive", failed_query)
    with pytest.raises(RuntimeError, match="Archive query failed"):
        example.load_gaia_astrometry(archive_path="unused.parquet")


@pytest.mark.parametrize("with_radar", [False, True])
def test_main_omits_gaia_fit_and_leaves_reference_loading_to_estimation(monkeypatch, with_radar):
    monkeypatch.setattr(example.spice, "load_standard_kernels", lambda: None)
    monkeypatch.setattr(example, "load_gaia_astrometry", lambda: None)
    optical, radar = [object()], [object()] if with_radar else []
    monkeypatch.setattr(example, "load_tracking_data", lambda: (optical, [], radar, []))
    monkeypatch.setattr(example, "observation_epoch_bounds", lambda tracks: (0., 10.))

    def unexpected_horizons_query(**kwargs):
        pytest.fail("main must leave reference loading to perform_estimation")

    monkeypatch.setattr(example, "HorizonsQuery", unexpected_horizons_query)
    calls, plotted = [], []

    def estimate(*args, **kwargs):
        assert args[3] is None  # Initial state is constructed in perform_estimation.
        assert args[7] is None  # No Gaia observations.
        assert "reference_state_history" not in kwargs
        calls.append(args[0])
        return None, None, None, None

    monkeypatch.setattr(example, "perform_estimation", estimate)
    monkeypatch.setattr(example, "print_residual_summary", lambda *args: None)
    monkeypatch.setattr(example, "print_orbit_difference_rsw", lambda *args: None)
    monkeypatch.setattr(example, "plot_diagnostics", lambda results: plotted.extend(results))
    example.main(False)
    expected = ["MPC astrometry"] + (["MPC astrometry and JPL radar"] if with_radar else [])
    assert plotted == expected
    assert calls == [optical] + ([optical + radar] if with_radar else [])


def test_estimation_loads_and_reuses_horizons_reference_for_initialization(monkeypatch):
    queries, environments = [], []
    epochs = np.arange(-10., 11.)
    states = np.tile([1.e11, 0., 0., 0., 30000., 0.], (len(epochs), 1))

    def query(**kwargs):
        queries.append(kwargs)
        return SimpleNamespace(cartesian=lambda **kwargs: np.column_stack((epochs, states)))

    class StopAfterEnvironment(Exception):
        pass

    def create_environment(first, final, gaia, histories, selected):
        environments.append(histories)
        raise StopAfterEnvironment

    monkeypatch.setattr(example, "HorizonsQuery", query)
    monkeypatch.setattr(example, "create_bodies", create_environment)
    example.load_reference_state_history.cache_clear()
    try:
        for _ in range(2):
            with pytest.raises(StopAfterEnvironment):
                example.perform_estimation([], [], 0., None, -10., 10., False)
        assert len(queries) == 1
        assert queries[0]["query_id"] == f"{example.TARGET};"
        assert queries[0]["location"] == example.HORIZONS_ORIGIN
        assert environments[0][example.TARGET] is environments[1][example.TARGET]
    finally:
        example.load_reference_state_history.cache_clear()


@pytest.mark.parametrize("apply_final_correction", [False, True])
def test_reported_parameters_match_evaluated_residuals(apply_final_correction):
    parameters = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
    output = SimpleNamespace(
        best_iteration=0,
        final_parameters=parameters[:, 0],
        parameter_history=parameters if apply_final_correction else parameters[:, :2],
        residual_history=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        active_flags_per_iteration=np.array([[True, True], [True, False], [True, True]]),
    )
    # Exclude inactive observations and the optional, unevaluated final update.
    np.testing.assert_allclose(example.last_iteration_residuals(output), [0.2, 0.6])
    np.testing.assert_allclose(example.last_iteration_parameters(output), [2.0, 4.0])


def test_residual_and_orbit_plots_use_last_iteration(monkeypatch):
    example.plt.close("all")
    reference = np.array([1.0e8, 0.0, 0.0, 0.0, 1000.0, 0.0])
    last_state = reference + [1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0]
    output = SimpleNamespace(
        best_iteration=0,
        final_residuals=np.array([0.1, 0.1]),
        residual_history=np.array([[0.1, 2.0], [0.1, 3.0]]),
        active_flags_per_iteration=np.ones((2, 2), dtype=bool),
        covariance=np.eye(6),
        simulation_results_per_iteration=[
            SimpleNamespace(dynamics_results=SimpleNamespace(state_history_float={0.0: state}))
            for state in (reference, last_state)
        ],
    )
    range_type = example.observable_models_setup.model_settings.n_way_range_type
    monkeypatch.setattr(example, "observation_scalar_data", lambda dataset: {
        "times": np.array([0.0, 1.0]),
        "weights": np.array([4.0, 9.0]),
        "observable_types": np.array([range_type, range_type], dtype=object),
        "components": np.zeros(2, dtype=int),
        "station_labels": np.array(["Test", "Test"]),
        "space_astrometry": np.zeros(2, dtype=bool),
    })
    covariance_calls = []

    def propagate_covariance(covariance, interface, epochs):
        covariance_calls.append(covariance)
        return {epoch: covariance for epoch in epochs}

    monkeypatch.setattr(example.estimation_analysis, "propagate_covariance", propagate_covariance)
    try:
        example.plot_residuals("test", output, None)
        figures = [example.plt.figure(n) for n in example.plt.get_fignums()]
        np.testing.assert_allclose(figures[0].axes[0].collections[0].get_offsets()[:, 1], [2, 3])
        np.testing.assert_allclose(figures[1].axes[0].collections[0].get_offsets()[:, 1], [4, 9])
        example.plt.close("all")

        example.plot_orbit_difference(
            "test", output, SimpleNamespace(state_transition_interface=None),
            np.array([0.0]), np.array([reference]),
        )
        figures = [example.plt.figure(n) for n in example.plt.get_fignums()]
        for axis, expected in zip(figures[0].axes, [1, 2, 3]):
            np.testing.assert_allclose(axis.lines[0].get_ydata(), [expected])
        assert covariance_calls[0] is output.covariance
    finally:
        example.plt.close("all")


def test_mpc_prefit_rejection_uses_each_components_weight(monkeypatch):
    obs = example.observations
    dataset = obs.ObservationDataset()
    residuals = [
        np.array([[5., 0.], [2.5, 0.], [2.50001, 0.], [0., -1.67], [0., 1.6], [100., 0.]]),
        np.array([[0., 6.]]),  # MPC space-based astrometry is also screened.
        np.array([[100., 100.]]),  # Gaia is excluded.
        np.array([[100.]]),  # Radar is excluded.
    ]
    for i, (receiver, observable) in enumerate([
        ("Earth", obs.angular_position_type), ("WISE", obs.angular_position_type),
        ("Gaia", obs.angular_position_type), ("Earth", obs.one_way_range_type),
    ]):
        link = obs.LinkDefinition({
            obs.transmitter: obs.LinkEndId(example.TARGET, ""),
            obs.receiver: obs.LinkEndId(receiver, ""),
        })
        dataset.add_observation_set(
            observable, link, np.zeros_like(residuals[i]),
            [1., 1., 2., 3., 4., 5.] if i == 0 else [1.], obs.receiver,
        )
    dataset.set_weight_vector_for_set(0, np.array([1., 1., 4., 9., 4., 9., 4., 9., 4., 9., 1., 1.]))
    q = obs.observation_query
    dataset.reject_observations((q.set_id == 0) & (q.time == 5.), "previous rejection")
    weights = dataset.get_weight_diagonal().copy()
    sentinel_bodies = object()

    def compute_prefit(data, simulators, bodies):
        assert data is dataset and bodies is sentinel_bodies
        for set_id, values in enumerate(residuals):
            data.set_residuals_for_set(set_id, values)

    monkeypatch.setattr(example.observations_simulation_settings, "create_observation_simulators", lambda *args: [])
    monkeypatch.setattr(obs, "compute_residuals_and_dependent_variables", compute_prefit)
    example.reject_mpc_prefit_outliers(dataset, sentinel_bodies)
    assert dataset.get_observation_ids(q.active & (q.set_id == 0)) == [0, 1, 4]
    assert not dataset.get_observation_ids(q.active & (q.set_id == 1))
    assert len(dataset.get_observation_ids(q.active & (q.set_id == 2))) == 1
    assert len(dataset.get_observation_ids(q.active & (q.set_id == 3))) == 1
    np.testing.assert_array_equal(dataset.get_weight_diagonal(), weights)
    for set_id, values in enumerate(residuals):
        np.testing.assert_array_equal(dataset.residuals_for_set(set_id), values)
