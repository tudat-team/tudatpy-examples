"""Targeted campaign checks: no multi-day propagation or orbit fit is performed."""

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
import pickle
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mro_tnf_estimation_test as campaign
import mro_campaign_queue as campaign_queue
import mro_campaign_primary_mask as primary_mask


def native_test_empirical_boundaries(case, arc_index=0):
    """Use a valid audited long-arc boundary list for edge-aware native tests."""
    if case.empirical_edge_policy in {
            "merge_case001_zero_edges", "merge_one_orbit_conservative_edges",
            "merge_one_orbit_h_zero_edges"}:
        setup = json.loads((
            campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
            / f"case_001_arc_{arc_index:02d}" / "setup.json"
        ).read_text())
        return setup["empirical_arc_start_times"]
    return [378681900., 378682000., 378682100.]


def test_original_dates_and_input_selection():
    """All seven original arcs have local inputs, including the preceding-day TNF."""
    assert len(campaign.ARCS) == 7
    for index, (start, end) in enumerate(campaign.ARCS):
        inputs = campaign.local_inputs(index)
        assert inputs[1:3] == [datetime.fromisoformat(start), datetime.fromisoformat(end)]
        assert inputs[3] and inputs[5] and inputs[6] and inputs[7]
    # The first pass starts in a file named for the preceding UTC day.
    assert any("2011_365" in name for name in campaign.local_inputs(0)[3])


def test_example_uses_fixed_arcs_and_inputs(monkeypatch):
    """The normal entry point uses explicit settings independent of campaign env."""
    import mro_tnf_estimation as example

    monkeypatch.setenv("MRO_INTEGRATOR_COEFFICIENT_SET", "rkf78")
    monkeypatch.setenv("MRO_MAXIMUM_ITERATIONS", "99")
    monkeypatch.setenv("MRO_OBSERVATION_SIGMA_HZ", "0.003")
    source = Path(example.__file__).read_text()
    assert "MroExampleSettings" not in source
    assert "os.environ.get" not in source
    assert "legacy_settings_from_environment" not in source
    assert list(map(list, example.ESTIMATION_ARCS)) == json.loads(
        (campaign.DEFAULT_ROOT / "case_044" / "arcs.json").read_text()
    )

    manifest = json.loads(
        (campaign.DEFAULT_ROOT / "case_044" / "input_manifest.json").read_text()
    )
    for arc_index, (start, end) in enumerate(example.ESTIMATION_ARCS):
        actual = example.prepare_arc_inputs(
            arc_index, datetime.fromisoformat(start), datetime.fromisoformat(end)
        )
        assert actual == tuple(campaign.local_inputs(arc_index)[3:])
        for item in actual:
            for path in item if isinstance(item, list) else [item]:
                assert str(Path(path).resolve()) in manifest


def test_example_two_orbit_boundaries_match_saved_reference():
    """The standalone schedule reproduces every audited zero-sensitivity merge."""
    import mro_tnf_estimation as example

    total_parameters = 0
    for arc_index in range(7):
        setup = json.loads(
            (
                campaign.DEFAULT_ROOT
                / "subarc_diagnostic"
                / "validation"
                / f"case_001_arc_{arc_index:02d}"
                / "setup.json"
            ).read_text()
        )
        actual = example.empirical_arc_starts(
            setup["empirical_arc_start_times"], arc_index
        )
        saved = pd.read_csv(
            campaign.DEFAULT_ROOT
            / "case_044"
            / "arcs"
            / f"arc_{arc_index:02d}"
            / "parameter_prior_metadata.csv"
        )
        expected = saved.loc[
            saved.name.str.startswith("empirical_"), "subarc_start_tdb"
        ].drop_duplicates()
        # CSV decimal round-tripping may move the final binary bit (~6e-8 s).
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)
        total_parameters += 7 + 6 * len(actual)
    assert total_parameters == 871


def test_nominal_edge_merge_rejects_unknown_arc():
    """The fixed edge merge applies only to the seven estimation arcs."""
    import mro_tnf_estimation as example

    starts = np.arange(20, dtype=float) * 100.0
    with pytest.raises(ValueError, match="seven estimation arcs"):
        example.empirical_arc_starts(starts, 7)


def test_retained_tag_mask_and_plain_result_are_picklable(monkeypatch):
    """Workers return primary-mask arrays only, never a native EstimationOutput."""
    import mro_tnf_estimation as example

    orbit = pd.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0], "R": [0.0] * 4, "T": [0.0] * 4, "N": [0.0] * 4}
    )
    residuals = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "spice": [0.0, 0.0],
            "prefit": [0.0, 0.0],
            "postfit": [0.0, 0.0],
        }
    )
    assert example.retained_observation_tag_orbit(orbit, residuals).t.tolist() == [
        1.0,
        2.0,
    ]
    monkeypatch.setattr(example, "compare_history_to_spice", lambda *args: orbit)

    class Output:
        best_iteration = 1
        parameter_history = np.array([[1.0, 1.5], [2.0, 2.5]])
        correlations = np.array([[1.0, -0.25], [-0.25, 1.0]])

    plain = example._plain_arc_result(
        {
            "arc_index": 0,
            "arc_bounds": (0.0, 3.0),
            "residuals": residuals,
            "prefit_state_history": {},
            "postfit_state_history": {},
            "estimation_output": Output(),
            "parameter_metadata": [
                {
                    "index": 0,
                    "name": "x",
                    "unit": "m",
                    "group": "state",
                    "subarc_start": None,
                },
                {
                    "index": 1,
                    "name": "sun_scale",
                    "unit": "1",
                    "group": "Sun",
                    "subarc_start": None,
                },
            ],
            "initial_parameters": np.array([1.0, 2.0]),
            "estimation_epoch": 1.5,
            "empirical_arc_start_times": [0.0],
            "inverse_apriori_covariance": np.eye(2),
        },
    )
    assert "estimation_output" not in plain
    assert plain["correlations"][0, 1] == -0.25
    assert plain["parameters"][0]["validity_start"] == 1.5
    assert plain["parameters"][0]["validity_end"] == 1.5
    assert plain["parameters"][1]["validity_start"] == 0.0
    assert plain["parameters"][1]["validity_end"] == 3.0
    pickle.dumps(plain)


def test_nominal_plots_have_units_steps_state_corrections_and_subset_ids(monkeypatch):
    """Headless plots preserve signed matrices and noncontiguous arc identities."""
    import mro_tnf_estimation as example

    monkeypatch.setattr(example.plt, "show", lambda: None)
    residuals = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "spice": [0.001, -0.001],
            "prefit": [0.002, -0.002],
            "postfit": [0.0005, -0.0005],
            "arc_index": [3, 3],
        }
    )
    orbit = pd.DataFrame(
        {
            "t": [1.0, 2.0],
            "R": [0.1, -0.1],
            "T": [0.2, -0.2],
            "N": [0.3, -0.3],
            "arc_index": [3, 3],
        }
    )
    parameters = []
    for index, (name, unit) in enumerate(
        zip(("x", "y", "z", "vx", "vy", "vz"), ("m", "m", "m", "m/s", "m/s", "m/s"))
    ):
        parameters.append(
            {
                "index": index,
                "name": name,
                "unit": unit,
                "group": "state",
                "subarc_start": None,
                "value": float(index),
                "delta": 0.1 * (index + 1),
                "arc_index": 3,
                "validity_start": 1.5,
                "validity_end": 1.5,
            }
        )
    parameters.extend(
        [
            {
                "index": 6,
                "name": "sun_scale",
                "unit": "1",
                "group": "Sun",
                "subarc_start": None,
                "value": 1.0,
                "delta": 0.0,
                "arc_index": 3,
                "validity_start": 0.0,
                "validity_end": 3.0,
            },
            {
                "index": 7,
                "name": "empirical_T_constant",
                "unit": "m/s^2",
                "group": "empirical 00",
                "subarc_start": 0.5,
                "value": 2.0e-8,
                "delta": 2.0e-8,
                "arc_index": 3,
                "validity_start": 0.5,
                "validity_end": 2.5,
            },
        ]
    )
    correlations = np.eye(len(parameters))
    correlations[0, 1] = correlations[1, 0] = -0.5
    result = {
        "arc_index": 3,
        "arc_bounds": (0.0, 3.0),
        "best_iteration": 2,
        "residuals": residuals,
        "prefit_orbit": orbit,
        "postfit_orbit": orbit,
        "parameters": parameters,
        "correlations": correlations,
    }
    figures = example.plot_results([result])
    assert len(figures) == 5
    coefficient_axes = figures[2].axes
    assert coefficient_axes[0].get_ylabel() == "scale [1]"
    assert coefficient_axes[1].get_ylabel() == "coefficient [m/s²]"
    np.testing.assert_allclose(
        coefficient_axes[0].lines[0].get_xdata(), np.array([0.0, 3.0]) / 86400.0
    )
    np.testing.assert_allclose(
        coefficient_axes[1].lines[0].get_xdata(), np.array([0.5, 2.5]) / 86400.0
    )
    np.testing.assert_allclose(
        coefficient_axes[1].lines[0].get_ydata(), [2.0e-8, 2.0e-8]
    )
    assert "not continuous states" in figures[3]._suptitle.get_text()
    assert figures[3].axes[0].get_ylabel() == "midpoint correction [m]"
    assert figures[3].axes[3].get_ylabel() == "midpoint correction [m/s]"
    correlation_image = figures[-1].axes[0].images[0]
    assert correlation_image.get_clim() == (-1.0, 1.0)
    np.testing.assert_array_equal(
        np.asarray(correlation_image.get_array()), result["correlations"]
    )
    for figure in figures:
        example.plt.close(figure)


def test_nominal_worker_has_no_campaign_settings_or_callbacks():
    """The standalone worker accepts just an arc ID, with no model switches."""
    import inspect
    import mro_tnf_estimation as example

    assert list(inspect.signature(example.run_arc).parameters) == ["arc_index"]


def test_nominal_parallel_entry_point_submits_all_arcs(monkeypatch):
    """The parent submits seven fixed fits and plots only their plain results."""
    import mro_tnf_estimation as example

    submitted = []
    plotted = []

    class Future:
        def __init__(self, arc_index):
            self.arc_index = arc_index

        def result(self):
            return {"arc_index": self.arc_index, "best_iteration": 4}

    class Pool:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 7
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, arc_index):
            assert function is example.run_arc
            submitted.append(arc_index)
            return Future(arc_index)

    monkeypatch.setattr(example, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(example, "as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr(example, "get_mro_files", lambda *args: None)
    monkeypatch.setattr(example, "_pooled_primary_metrics", lambda results: dict.fromkeys(
        ["residual_rms_mhz", "residual_max_mhz", "R_rms_m", "T_rms_m", "N_rms_m",
         "position_rms_m"], 0.0
    ))
    monkeypatch.setattr(example, "plot_results", plotted.extend)
    monkeypatch.setattr(example.plt, "show", lambda: None)
    example.run_estimation()

    # Completion order must not scramble arc identities in the parent plots.
    assert submitted == list(range(7))
    assert [result["arc_index"] for result in plotted] == list(range(7))


def test_nominal_orbit_difference_uses_spice_reference_rtn(monkeypatch):
    """The nominal score rotates Cartesian error with the reference state."""
    import mro_tnf_estimation as example
    import spiceypy

    reference_si = np.array([1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0])
    estimated = reference_si + np.array([2.0, 3.0, 4.0, 0.0, 0.0, 0.0])

    class Interpolation:
        @staticmethod
        def interpolate(epoch):
            return estimated

    monkeypatch.setattr(
        example.interpolators,
        "create_one_dimensional_vector_interpolator",
        lambda *args, **kwargs: Interpolation(),
    )
    monkeypatch.setattr(
        spiceypy,
        "spkezr",
        lambda *args, **kwargs: (reference_si / 1000.0, 0.0),
    )
    history = {-600.0: estimated, 600.0: estimated}
    result = example.compare_history_to_spice(history, 0.0, 0.0, 60.0)
    np.testing.assert_allclose(result[["R", "T", "N"]], [[2.0, 3.0, 4.0]])


@pytest.mark.parametrize("case", list(campaign.cases().values()))
def test_case_invariants(case):
    """Every proposed case obeys the user's parameter-redundancy and shadowing rules."""
    case.validate()
    env = case.environment()
    assert all(env[key] == "0" for key in env if "SHADOWING" in key)
    if "arcwise" in (case.drag_scale, case.lift_scale):
        assert "T" not in case.empirical_components
    if not case.lift:
        assert env["MRO_LIFT_COEFFICIENT"] == "0" and case.lift_scale == "fixed"


@pytest.mark.parametrize("changes", [
    {"lift": False}, {"drag_scale": "arcwise"}, {"step_seconds": 0},
    {"scale_sigma": float("nan")}, {"drag_scale_sigma": 0.0},
    {"aerodynamic_model": "sentman"},
    {"integrator": "rkdp87"},
    {"sun_radiation_shadowing_pixels": 10},
    {"mars_radiation_shadowing_pixels": 20},
    {"empirical_components": "RR"}, {"normal_empirical_sigma_m_s2": -1e-8},
    {"position_sigma_m": 99.9}, {"velocity_sigma_m_s": 0.099},
])
def test_bad_configurations_fail_early(changes):
    """Invalid settings fail before starting expensive workers."""
    with pytest.raises(ValueError):
        replace(campaign.Case(), **changes).validate()


def test_shadowing_controls_are_source_specific_and_fixed_at_twenty_pixels():
    """Future final-phase controls cannot couple Sun, aero, and Mars shadows."""
    case = replace(
        campaign.Case(), mars_radiation_target="panelled",
        sun_radiation_shadowing_pixels=20,
        aerodynamic_shadowing_pixels=0,
        mars_radiation_shadowing_pixels=20,
    )
    case.validate()
    environment = case.environment()
    assert environment["MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS"] == "20"
    assert environment["MRO_AERODYNAMIC_SELF_SHADOWING_PIXELS"] == "0"
    assert environment["MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS"] == "20"
    assert environment["MRO_RADIATION_SELF_SHADOWING_PIXELS"] == "0"


def test_metrics_pool_squared_samples():
    """Global RMS uses sample sums, not averages of per-arc RMS values."""
    residuals = pd.DataFrame({"spice": [0.001, 0.003], "postfit": [0.002, 0.004]})
    orbit = pd.DataFrame({"R": [3., 0.], "T": [0., 4.], "N": [0., 0.]})
    result = campaign.metrics(residuals, orbit)
    assert result["residual_rms_mhz"] == pytest.approx(np.sqrt(10))
    assert result["position_rms_m"] == pytest.approx(np.sqrt(12.5))
    assert result["position_max_m"] == 4


@pytest.fixture(scope="module")
def kernels():
    import mro_tnf_estimation as example
    inputs = campaign.local_inputs(0)
    example.load_spice_kernels(inputs[4], inputs[5], inputs[8], inputs[9], inputs[10])
    return example


@pytest.fixture(scope="module")
def environments(kernels):
    """Keep at most one environment alive; MCD and high-degree models use substantial RAM."""
    from tudatpy.astro.time_representation import Time
    cache = {}
    previous = dict(os.environ)
    try:
        def get_environment(case):
            key = (
                case.lift, case.aerodynamic_model, case.reduced_solar_arrays,
                case.mars_radiation_target, case.sun_radiation_shadowing_pixels,
                case.aerodynamic_shadowing_pixels,
                case.mars_radiation_shadowing_pixels,
                case.mcd_scenario, case.mcd_high_resolution,
            )
            if key in cache:
                return cache[key]
            cache.clear()
            os.environ.update(case.environment())
            # The Earth rotation splines need several hourly samples, even for a short smoke test.
            cache[key] = kernels.create_environment(Time(378671000.), Time(378693000.))
            return cache[key]
        yield get_environment
    finally:
        os.environ.clear()
        os.environ.update(previous)


@pytest.mark.parametrize("number,case", list(campaign.cases().items()))
def test_native_parameters_priors_and_partials(number, case, environments, kernels, monkeypatch):
    """Construct native estimator/partials for every supported parameter and force setup."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    from tudatpy.estimation.estimation_analysis import Estimator
    if case.blocked_reason:
        pytest.skip(case.blocked_reason)
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(case)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.), Time(378681900.), Time(378682100.))
    plan = campaign.ParameterPlan(case, arc_index=0)
    parameter_settings = plan.settings(
        settings, bodies, native_test_empirical_boundaries(case)
    )
    parameters = parameters_setup.create_parameter_set(parameter_settings, bodies, settings)
    inverse_prior = plan.priors(parameters)
    table = pd.DataFrame(plan.rows)
    # Native block ordering determines both the precision matrix and the plot labels.
    assert table["index"].tolist() == list(range(parameters.parameter_set_size))
    np.testing.assert_allclose(np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2)
    np.testing.assert_allclose(
        np.diag(inverse_prior)[:6],
        [case.position_sigma_m ** -2] * 3 + [case.velocity_sigma_m_s ** -2] * 3,
    )
    assert np.count_nonzero(inverse_prior - np.diag(np.diag(inverse_prior))) == 0
    values = parameters.parameter_vector.copy()
    empirical = table[table.name.str.startswith("empirical_")]
    if len(empirical):
        # This nontrivial shape-first order is the C++ convention used by the plots.
        first = empirical[empirical.subarc_start_tdb == plan.arc_times[0]]
        expected = [f"empirical_{c}_{shape}" for shape in ("constant", "sine", "cosine")
                    if shape in case.empirical_shapes for c in "RTN" if c in case.empirical_components]
        assert first.name.tolist() == expected
    if number == "001":
        assert case.drag_scale == case.lift_scale == "fixed"
        assert "drag_scale" not in set(table.name)
        assert "lift_scale" not in set(table.name)
        assert "sun_scale" in set(table.name)
        assert not table.prior_sigma.isna().any()
    Estimator(bodies, parameters, [], settings, integrate_on_creation=False)
    np.testing.assert_array_equal(parameters.parameter_vector, values)


def test_nominal_case044_native_parameter_prior_maps_all_arcs(
    environments, kernels, monkeypatch
):
    """Construct native standalone blocks and match all saved case044 maps."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    from tudatpy.estimation.estimation_analysis import Estimator

    case = campaign.cases()["044"]
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(case)
    propagator, _ = kernels.create_propagator_settings(
        bodies,
        Time(378682000.0),
        Time(378681900.0),
        Time(378682100.0),
    )
    total_parameters = 0
    for arc_index in range(7):
        setup = json.loads(
            (
                campaign.DEFAULT_ROOT
                / "subarc_diagnostic"
                / "validation"
                / f"case_001_arc_{arc_index:02d}"
                / "setup.json"
            ).read_text()
        )
        starts = kernels.empirical_arc_starts(
            setup["empirical_arc_start_times"], arc_index
        )
        native_settings = kernels.create_parameter_settings(
            propagator, bodies, starts
        )
        parameters = parameters_setup.create_parameter_set(
            native_settings, bodies, propagator
        )
        metadata = kernels.create_parameter_metadata(
            parameters, starts
        )
        inverse_prior = kernels.create_inverse_apriori_covariance(
            parameters
        )
        saved = pd.read_csv(
            campaign.DEFAULT_ROOT
            / "case_044"
            / "arcs"
            / f"arc_{arc_index:02d}"
            / "parameter_prior_metadata.csv"
        )
        assert [row["name"] for row in metadata] == saved.name.tolist()
        assert [row["index"] for row in metadata] == list(
            range(parameters.parameter_set_size)
        )
        assert not {"drag_scale", "lift_scale"} & {
            row["name"] for row in metadata
        }
        np.testing.assert_allclose(
            np.diag(inverse_prior),
            saved.prior_sigma.to_numpy() ** -2,
            rtol=0.0,
            atol=0.0,
        )
        empirical = saved[saved.name.str.startswith("empirical_")]
        np.testing.assert_allclose(
            starts,
            empirical.subarc_start_tdb.drop_duplicates(),
            rtol=0.0,
            atol=1.0e-7,
        )
        Estimator(
            bodies, parameters, [], propagator, integrate_on_creation=False
        )
        total_parameters += parameters.parameter_set_size
    assert total_parameters == 871


def test_prior_mapping_rejects_missing_metadata():
    """An unrecognized parameter cannot silently receive zero prior precision."""
    class UnassignedParameter:
        parameter_set_size = 3
    with pytest.raises(ValueError, match="Unassigned"):
        campaign.ParameterPlan(campaign.Case()).priors(UnassignedParameter())


def test_normal_only_prior_override_preserves_t_and_has_no_orphan_scales(environments, kernels, monkeypatch):
    """The final fallback tightens N only; T and complete native index coverage stay explicit."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    case = replace(
        campaign.cases()["001"], empirical_sigma_m_s2=1e-7,
        constant_empirical_sigma_m_s2=1e-7, normal_empirical_sigma_m_s2=1e-8)
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(case)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.), Time(378681900.), Time(378682100.))
    plan = campaign.ParameterPlan(case, arc_index=0)
    parameters = parameters_setup.create_parameter_set(
        plan.settings(settings, bodies, [378681900., 378682000., 378682100.]), bodies, settings)
    inverse_prior = plan.priors(parameters)
    table = pd.DataFrame(plan.rows)
    empirical = table[table.name.str.startswith("empirical_")]
    assert set(empirical[empirical.name.str.startswith("empirical_T_")].prior_sigma) == {1e-7}
    assert set(empirical[empirical.name.str.startswith("empirical_N_")].prior_sigma) == {1e-8}
    assert not empirical.name.str.startswith("empirical_R_").any()
    assert "drag_scale" not in set(table.name) and "lift_scale" not in set(table.name)
    assert "sun_scale" in set(table.name)
    assert table["index"].tolist() == list(range(parameters.parameter_set_size))
    np.testing.assert_allclose(np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2)


def test_edge_merge_uses_exact_case001_zero_boundaries():
    case = campaign.cases()["002"]
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    expected = {
        0: {378898756.90110093, 378912224.0540009},
        5: {379962503.7045312},
    }
    for arc in range(7):
        setup = json.loads(
            (validation / f"case_001_arc_{arc:02d}" / "setup.json").read_text()
        )
        original = setup["empirical_arc_start_times"]
        uniform = campaign.empirical_arc_times(
            replace(case, empirical_edge_policy="uniform"), original, arc
        )
        merged = campaign.empirical_arc_times(case, original, arc)
        assert merged[0] == original[0] == uniform[0]
        assert set(uniform) - set(merged) == expected.get(arc, set())
        assert len(merged) == 20 - len(expected.get(arc, set()))


def test_edge_merge_native_prior_indices_cover_115_and_121_parameters(
        environments, kernels, monkeypatch):
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    case = campaign.cases()["002"]
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(case)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.), Time(378681900.), Time(378682100.))
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    for arc, expected_count in ((0, 115), (5, 121)):
        original = json.loads(
            (validation / f"case_001_arc_{arc:02d}" / "setup.json").read_text()
        )["empirical_arc_start_times"]
        plan = campaign.ParameterPlan(case, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings)
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        assert parameters.parameter_set_size == expected_count
        assert table["index"].tolist() == list(range(expected_count))
        np.testing.assert_allclose(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2)
        assert plan.arc_times[0] == original[0]
        removed = set(campaign.empirical_arc_times(
            replace(case, empirical_edge_policy="uniform"), original, arc
        )) - set(plan.arc_times)
        assert not table.subarc_start_tdb.isin(removed).any()


def test_case020_one_orbit_edges_support_and_native_priors_all_arcs(
        kernels, monkeypatch):
    """Case020 derives its own one-orbit topology; no two-orbit index is reused."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    case = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_020.json"
    )
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_019" / "settings.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(case)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"empirical_periods", "empirical_edge_policy"}
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)

    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    expected_counts = [228, 246, 240, 246, 240, 234, 240]
    # Independently documented conservative allowance: compression interval,
    # two-way light-time bound, and four-step interpolation support.
    support_margins = {
        0: 1216.303, 1: 1190.093, 2: 1163.578, 3: 1138.206,
        4: 1112.664, 5: 1086.567, 6: 1063.993,
    }
    for arc, expected_count in enumerate(expected_counts):
        setup = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())
        original = setup["empirical_arc_start_times"]
        removed_indices = campaign.ONE_ORBIT_EDGE_MERGE_BOUNDARY_INDICES.get(
            arc, ()
        )
        expected_boundaries = [
            epoch for index, epoch in enumerate(original)
            if index not in removed_indices
        ]
        assert campaign.empirical_arc_times(case, original, arc) == expected_boundaries
        assert expected_boundaries[0] == original[0]

        lower = setup["observation_epoch_min_tdb"] - support_margins[arc]
        upper = setup["observation_epoch_max_tdb"] + support_margins[arc]
        propagation_end = setup["propagation_bounds"][1]
        for index in removed_indices:
            block_start = original[index]
            block_end = (
                original[index + 1] if index + 1 < len(original)
                else propagation_end
            )
            assert block_end < lower or block_start > upper

        bodies = kernels.create_environment(
            Time(378671000.), Time(378693000.))[0]
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        plan = campaign.ParameterPlan(case, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        assert parameters.parameter_set_size == expected_count
        assert table["index"].tolist() == list(range(expected_count))
        assert plan.arc_times == expected_boundaries
        assert "sun_scale" not in set(table.name)
        empirical = table[table.name.str.startswith("empirical_")]
        assert set(empirical.name) == {
            f"empirical_{component}_{shape}"
            for component in "TN" for shape in ("constant", "sine", "cosine")
        }
        assert set(empirical.prior_sigma) == {3.0e-6}
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
    assert sum(expected_counts) == 1674


def test_case021_one_orbit_arc_scales_and_native_priors_all_arcs(
        kernels, monkeypatch):
    """Case021 changes only the audited timescale topology from case012."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    case = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_021.json"
    )
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_012" / "settings.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(case)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"empirical_periods", "empirical_edge_policy"}
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)

    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    expected_counts = [192, 207, 202, 207, 202, 197, 202]
    support_margins = {
        0: 1216.303, 1: 1190.093, 2: 1163.578, 3: 1138.206,
        4: 1112.664, 5: 1086.567, 6: 1063.993,
    }
    for arc, expected_count in enumerate(expected_counts):
        setup = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())
        original = setup["empirical_arc_start_times"]
        removed_indices = campaign.ONE_ORBIT_EDGE_MERGE_BOUNDARY_INDICES.get(
            arc, ()
        )
        expected_boundaries = [
            epoch for index, epoch in enumerate(original)
            if index not in removed_indices
        ]
        lower = setup["observation_epoch_min_tdb"] - support_margins[arc]
        upper = setup["observation_epoch_max_tdb"] + support_margins[arc]
        propagation_end = setup["propagation_bounds"][1]
        for index in removed_indices:
            start = original[index]
            end = original[index + 1] if index + 1 < len(original) else propagation_end
            assert end < lower or start > upper

        bodies = kernels.create_environment(
            Time(378671000.), Time(378693000.))[0]
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        plan = campaign.ParameterPlan(case, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        assert plan.arc_times == expected_boundaries
        assert parameters.parameter_set_size == expected_count
        assert table["index"].tolist() == list(range(expected_count))
        assert not table.name.str.startswith("empirical_T_").any()
        assert set(table.loc[table.name == "drag_scale", "prior_sigma"]) == {0.2}
        assert set(table.loc[table.name == "lift_scale", "prior_sigma"]) == {0.2}
        empirical = table[table.name.str.startswith("empirical_N_")]
        assert set(empirical.name) == {
            "empirical_N_constant", "empirical_N_sine", "empirical_N_cosine"
        }
        assert set(empirical.prior_sigma) == {3.0e-6}
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
    assert sum(expected_counts) == 1409


@pytest.mark.parametrize(
    "case_id,control_id,expected_counts,expected_total,block_width",
    [
        ("058", "019", [222, 240, 240, 240, 240, 222, 240], 1644, 6),
        ("059", "012", [187, 202, 202, 202, 202, 187, 202], 1384, 5),
    ],
)
def test_h_zero_corrected_one_orbit_boundaries_and_native_priors_all_arcs(
        kernels, monkeypatch, case_id, control_id, expected_counts,
        expected_total, block_width):
    """Corrected one-orbit cases use the exact selected-H edge removals."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    case = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / f"case_{control_id}" / "settings.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(case)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"empirical_periods", "empirical_edge_policy"}
    assert case.empirical_edge_policy == "merge_one_orbit_h_zero_edges"
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)

    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    counts = []
    for arc, expected_count in enumerate(expected_counts):
        original = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())["empirical_arc_start_times"]
        removed_indices = (
            campaign.ONE_ORBIT_H_ZERO_EDGE_MERGE_BOUNDARY_INDICES.get(arc, ())
        )
        expected_boundaries = [
            epoch for index, epoch in enumerate(original)
            if index not in removed_indices
        ]
        assert campaign.empirical_arc_times(case, original, arc) == expected_boundaries
        # The propagation-start lookup anchor is retained, especially on arc 05.
        assert expected_boundaries[0] == original[0]

        # Arc-wise scale parameters install time-varying functions on the body;
        # use the same fresh-environment isolation as real arc workers.
        bodies = kernels.create_environment(
            Time(378671000.), Time(378693000.))[0]
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        plan = campaign.ParameterPlan(case, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        counts.append(parameters.parameter_set_size)
        assert parameters.parameter_set_size == expected_count
        assert table["index"].tolist() == list(range(expected_count))
        assert plan.arc_times == expected_boundaries
        assert len(plan.arc_times) * block_width + (6 if case_id == "058" else 7) == expected_count
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
        assert not table.subarc_start_tdb.isin(
            [original[index] for index in removed_indices]
        ).any()
        if case_id == "058":
            assert "sun_scale" not in set(table.name)
            empirical = table[table.name.str.startswith("empirical_")]
            assert set(empirical.name) == {
                f"empirical_{component}_{shape}"
                for component in "TN"
                for shape in ("constant", "sine", "cosine")
            }
            assert set(empirical.prior_sigma) == {3.0e-6}
        else:
            assert not table.name.str.startswith("empirical_T_").any()
            assert set(table.loc[table.name == "drag_scale", "prior_sigma"]) == {0.2}
            assert set(table.loc[table.name == "lift_scale", "prior_sigma"]) == {0.2}
            empirical = table[table.name.str.startswith("empirical_N_")]
            assert set(empirical.name) == {
                "empirical_N_constant", "empirical_N_sine", "empirical_N_cosine"
            }
            assert set(empirical.prior_sigma) == {3.0e-6}
    assert counts == expected_counts
    assert sum(counts) == expected_total
    queue = json.loads((campaign.DEFAULT_ROOT / "RUN_QUEUE.json").read_text())
    job = next(item for item in queue["jobs"] if item["case_id"] == case_id)
    validated, reference, _, _ = campaign_queue.validate_job(
        campaign.DEFAULT_ROOT, job
    )
    assert validated == case
    assert reference.name == f"case_{control_id}"


def test_shape_cases_native_prior_indices_cover_all_arcs(
        environments, kernels, monkeypatch):
    """Approved case008/009 shapes have exact all-arc counts and prior coverage."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    expected_totals = {"008": 597, "009": 323}
    expected_by_arc = {
        "008": [79, 87, 87, 87, 87, 83, 87],
        "009": [43, 47, 47, 47, 47, 45, 47],
    }
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    for case_id in ("008", "009"):
        case = campaign.load_case(
            campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
        )
        for key, value in case.environment().items():
            monkeypatch.setenv(key, value)
        bodies = environments(case)
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        counts = []
        for arc in range(7):
            original = json.loads((
                validation / f"case_001_arc_{arc:02d}" / "setup.json"
            ).read_text())["empirical_arc_start_times"]
            plan = campaign.ParameterPlan(case, arc)
            parameters = parameters_setup.create_parameter_set(
                plan.settings(settings, bodies, original), bodies, settings)
            inverse_prior = plan.priors(parameters)
            table = pd.DataFrame(plan.rows)
            counts.append(parameters.parameter_set_size)
            assert table["index"].tolist() == list(
                range(parameters.parameter_set_size)
            )
            np.testing.assert_allclose(
                np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2)
            empirical = table[table.name.str.startswith("empirical_")]
            assert set(empirical.prior_sigma) == {3.0e-6}
        assert counts == expected_by_arc[case_id]
        assert sum(counts) == expected_totals[case_id]


def test_arcwise_drag_n_and_rn_cases_share_audited_boundaries_and_priors(
        kernels, monkeypatch):
    """Cases 011/013 replace T with drag and add only R in the paired case."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    expected_counts = {
        "011": [79, 87, 87, 87, 87, 83, 87],
        "013": [133, 147, 147, 147, 147, 140, 147],
    }
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    for case_id in ("011", "013"):
        case = campaign.load_case(
            campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
        )
        for key, value in case.environment().items():
            monkeypatch.setenv(key, value)
        counts = []
        for arc in range(7):
            # Creating an arc-wise scaling parameter installs its lookup on the
            # body, so each native construction needs a fresh body system.
            bodies = kernels.create_environment(
                Time(378671000.), Time(378693000.))[0]
            settings, _ = kernels.create_propagator_settings(
                bodies, "MRO", "Mars", Time(378682000.),
                Time(378681900.), Time(378682100.))
            original = json.loads((
                validation / f"case_001_arc_{arc:02d}" / "setup.json"
            ).read_text())["empirical_arc_start_times"]
            plan = campaign.ParameterPlan(case, arc)
            parameters = parameters_setup.create_parameter_set(
                plan.settings(settings, bodies, original), bodies, settings
            )
            inverse_prior = plan.priors(parameters)
            table = pd.DataFrame(plan.rows)
            counts.append(parameters.parameter_set_size)
            assert table["index"].tolist() == list(
                range(parameters.parameter_set_size)
            )
            np.testing.assert_array_equal(
                np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
            )
            assert not table.name.str.startswith("empirical_T_").any()
            assert "lift_scale" not in set(table.name)
            assert "sun_scale" in set(table.name)
            drag = table[table.name == "drag_scale"]
            empirical = table[table.name.str.startswith("empirical_")]
            assert drag.subarc_start_tdb.tolist() == plan.arc_times
            assert set(empirical.subarc_start_tdb) == set(plan.arc_times)
            assert set(drag.prior_sigma) == {0.2}
            assert set(empirical.prior_sigma) == {3.0e-6}
            expected_names = (
                {"empirical_N_constant", "empirical_N_sine", "empirical_N_cosine"}
                if case_id == "011" else
                {f"empirical_{component}_{shape}"
                 for component in "RN" for shape in ("constant", "sine", "cosine")}
            )
            assert set(empirical.name) == expected_names
        assert counts == expected_counts[case_id]


def test_case051_position_seed_perturbation_is_exact_case049_diagnostic(
        environments, kernels, monkeypatch):
    """The diagnostic changes only its seed/prior centre, not empirical blocks."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    import mro_tnf_estimation as example
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_049" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_051.json"
    )
    control_fields = asdict(control)
    candidate_fields = asdict(candidate)
    changed = {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    }
    assert changed == {"initial_position_offset_m"}
    assert candidate.initial_position_offset_m == [1.0, 1.0, 1.0]
    assert candidate.environment()["MRO_INITIAL_POSITION_OFFSET_M"] == "1.0,1.0,1.0"
    seed = np.array([10., 20., 30., 1., 2., 3.])
    shifted = example.apply_initial_position_offset(seed, "1,1,1")
    np.testing.assert_array_equal(seed, [10., 20., 30., 1., 2., 3.])
    np.testing.assert_array_equal(shifted, [11., 21., 31., 1., 2., 3.])
    np.testing.assert_array_equal(
        example.apply_initial_position_offset(seed, "0,0,0"), seed
    )
    with pytest.raises(ValueError, match="three finite"):
        example.apply_initial_position_offset(seed, "1,nan,1")

    queue = json.loads((campaign.DEFAULT_ROOT / "RUN_QUEUE.json").read_text())
    job = next(item for item in queue["jobs"] if item["case_id"] == "051")
    validated, reference, _, _ = campaign_queue.validate_job(
        campaign.DEFAULT_ROOT, job
    )
    validated.validate()
    assert validated.initial_position_offset_m == [1.0, 1.0, 1.0]
    assert reference.name == "case_049"

    # The unperturbed case-049 state must reproduce every persisted one-orbit
    # boundary bit-for-bit; edge merging, native parameter ordering, and every
    # prior index must then be identical to the completed control.
    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(candidate)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.),
        Time(378681900.), Time(378682100.))
    mars_mu = bodies.get("Mars").gravitational_parameter
    validation_root = (
        campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    )
    for arc in range(7):
        control_table = pd.read_csv(
            campaign.DEFAULT_ROOT / "case_049" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        reference_state = control_table.loc[:5, "nominal"].to_numpy()
        setup = json.loads((
            validation_root / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())
        raw_boundaries = example.empirical_arc_starts_from_reference_state(
            reference_state, mars_mu, *setup["propagation_bounds"]
        )
        np.testing.assert_array_equal(
            raw_boundaries, setup["empirical_arc_start_times"]
        )

        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, raw_boundaries), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        candidate_table = pd.DataFrame(plan.rows)
        columns = [
            "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
            "parameter_type",
        ]
        pd.testing.assert_frame_equal(
            candidate_table[columns].reset_index(drop=True),
            control_table[columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior),
            control_table.prior_sigma.to_numpy() ** -2,
        )


@pytest.mark.parametrize("offset", campaign.REQUESTED_POSITIVE_SEED_OFFSETS_M)
def test_requested_positive_seed_matrix_variants_are_single_field(offset):
    """The future shortlist matrix is ready but does not launch any fit."""
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_019" / "settings.json"
    )
    candidate = campaign.seed_offset_variant(control, offset)
    candidate.validate()
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"initial_position_offset_m"}
    assert candidate.initial_position_offset_m == (offset, offset, offset)
    assert candidate.environment()["MRO_INITIAL_POSITION_OFFSET_M"] == (
        f"{offset},{offset},{offset}"
    )
    assert candidate.apply_apriori_parameter_deviation
    assert candidate.velocity_sigma_m_s == control.velocity_sigma_m_s
    assert candidate.empirical_periods == control.empirical_periods
    assert candidate.empirical_edge_policy == control.empirical_edge_policy


def test_requested_seed_matrix_rejects_unapproved_offsets_and_shifted_controls():
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_019" / "settings.json"
    )
    with pytest.raises(ValueError, match="one of"):
        campaign.seed_offset_variant(control, 15.0)
    with pytest.raises(ValueError, match="zero-offset"):
        campaign.seed_offset_variant(
            replace(control, initial_position_offset_m=(1.0, 1.0, 1.0)), 2.5
        )


SEED_MATRIX_CASES = [
    ("060", "018", 1.0), ("061", "033", 1.0), ("062", "019", 1.0),
    ("063", "018", 2.5), ("064", "033", 2.5), ("065", "019", 2.5),
    ("066", "018", 10.0), ("067", "033", 10.0), ("068", "019", 10.0),
    ("069", "018", 25.0), ("070", "033", 25.0), ("071", "019", 25.0),
    ("072", "018", 100.0), ("073", "033", 100.0), ("074", "019", 100.0),
]


@pytest.mark.parametrize("case_id,control_id,offset", SEED_MATRIX_CASES)
def test_selected_seed_matrix_configs_and_queue_change_only_position_seed(
        case_id, control_id, offset):
    """All 15 approved variants are exact controls except +[d,d,d] m."""
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / f"case_{control_id}" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"initial_position_offset_m"}
    assert candidate.initial_position_offset_m == [offset, offset, offset]
    assert candidate.environment()["MRO_INITIAL_POSITION_OFFSET_M"] == (
        f"{offset},{offset},{offset}"
    )
    queue = json.loads((campaign.DEFAULT_ROOT / "RUN_QUEUE.json").read_text())
    job = next(item for item in queue["jobs"] if item["case_id"] == case_id)
    validated, reference, _, _ = campaign_queue.validate_job(
        campaign.DEFAULT_ROOT, job
    )
    assert validated == candidate
    assert reference.name == f"case_{control_id}"


@pytest.mark.parametrize(
    "case_id,control_id", [("060", "018"), ("061", "033"), ("062", "019")]
)
def test_selected_seed_matrix_native_boundaries_parameters_and_priors_match_control(
        environments, kernels, monkeypatch, case_id, control_id):
    """Offsets never alter the fixed control boundaries or prior/index tables."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    import mro_tnf_estimation as example

    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(candidate)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.),
        Time(378681900.), Time(378682100.))
    mars_mu = bodies.get("Mars").gravitational_parameter
    validation_root = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    for arc in range(7):
        control_table = pd.read_csv(
            campaign.DEFAULT_ROOT / f"case_{control_id}" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        reference_state = control_table.loc[:5, "nominal"].to_numpy()
        setup = json.loads((
            validation_root / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())
        raw_boundaries = example.empirical_arc_starts_from_reference_state(
            reference_state, mars_mu, *setup["propagation_bounds"]
        )
        np.testing.assert_array_equal(
            raw_boundaries, setup["empirical_arc_start_times"]
        )
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, raw_boundaries), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        candidate_table = pd.DataFrame(plan.rows)
        columns = [
            "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
            "parameter_type",
        ]
        pd.testing.assert_frame_equal(
            candidate_table[columns].reset_index(drop=True),
            control_table[columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior), control_table.prior_sigma.to_numpy() ** -2
        )


@pytest.mark.parametrize("case_id,offset", [
    ("075", 1.0), ("076", 10.0), ("077", 100.0),
])
def test_unconstrained_state_variants_change_only_seed_and_state_prior_mode(
        case_id, offset):
    """The same-objective trio is exact case018 apart from seed and zero state prior."""
    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_018" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"initial_position_offset_m", "constrain_initial_state_prior"}
    assert candidate.initial_position_offset_m == [offset, offset, offset]
    assert candidate.constrain_initial_state_prior is False


def test_unconstrained_state_native_all_arc_prior_maps_are_exact_case018_except_state(
        environments, kernels, monkeypatch):
    """All six physical state-prior rows/columns are exactly zero on all arcs."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup
    import mro_tnf_estimation as example

    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_075.json"
    )
    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(candidate)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.),
        Time(378681900.), Time(378682100.))
    mars_mu = bodies.get("Mars").gravitational_parameter
    validation_root = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    total = 0
    for arc in range(7):
        control_table = pd.read_csv(
            campaign.DEFAULT_ROOT / "case_018" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        setup = json.loads((
            validation_root / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())
        reference_state = control_table.loc[:5, "nominal"].to_numpy()
        raw_boundaries = example.empirical_arc_starts_from_reference_state(
            reference_state, mars_mu, *setup["propagation_bounds"]
        )
        np.testing.assert_array_equal(
            raw_boundaries, setup["empirical_arc_start_times"]
        )
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, raw_boundaries), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        total += parameters.parameter_set_size
        assert parameters.parameter_set_size == len(control_table)
        assert table["index"].tolist() == list(range(len(table)))
        assert table.loc[:5, "name"].tolist() == ["x", "y", "z", "vx", "vy", "vz"]
        assert not table.loc[:5, "prior_constrained"].any()
        assert table.loc[:5, "prior_sigma"].isna().all()
        np.testing.assert_array_equal(
            table.loc[:5, "prior_information_diagonal"], np.zeros(6)
        )
        np.testing.assert_array_equal(inverse_prior[:6, :], 0.0)
        np.testing.assert_array_equal(inverse_prior[:, :6], 0.0)
        assert table.loc[6:, "prior_constrained"].all()
        pd.testing.assert_frame_equal(
            table.loc[6:, ["index", "name", "unit", "subarc_start_tdb", "prior_sigma"]]
                .reset_index(drop=True),
            control_table.loc[6:, ["index", "name", "unit", "subarc_start_tdb", "prior_sigma"]]
                .reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior)[6:],
            control_table.loc[6:, "prior_sigma"].to_numpy() ** -2,
        )
    assert total == 1282


def test_unconstrained_state_matrix_and_iteration_diagnostics_serialize_without_fake_pull(
        tmp_path, monkeypatch):
    """Zero state precision is preserved and excluded from normalized updates."""
    class MatrixOutput:
        normalized_design_matrix = np.eye(2)
        normalization_terms = np.array([2., 3.])
        inverse_normalized_covariance = np.diag([1., 2.])
        residual_history = np.array([[1.], [-1.]])
        parameter_history = np.zeros((2, 2))
        correlations = np.eye(2)
        best_iteration = 0
        exception_during_inversion = False
        exception_during_propagation = False

    metadata = pd.DataFrame({
        "index": [0, 1], "name": ["x", "empirical_N_constant"],
        "unit": ["m", "m/s^2"], "subarc_start_tdb": [np.nan, 0.],
        "prior_sigma": [None, 1. / 3.],
        "prior_constrained": [False, True],
        "prior_information_diagonal": [0., 9.],
    })
    observations = pd.DataFrame({
        "time": [10., 20.], "link_id": [3, 3],
        "link_ends": ["A - A", "A - A"],
        "msrType": ["doppler", "doppler"], "spice": [.01, -.01],
    })
    matrix_directory = tmp_path / "matrix"
    matrix_directory.mkdir()
    summary = campaign.save_estimation_diagnostics(
        MatrixOutput(), observations, metadata, np.ones(2),
        np.diag([0., 9.]), matrix_directory, anchored_priors=True,
    )
    assert summary["unconstrained_parameter_count"] == 1
    assert summary["unconstrained_parameters"] == ["0:x@nan"]
    assert summary["unconstrained_prior_rows_and_columns_exact_zero"]
    matrices = np.load(matrix_directory / "normal_matrix_diagnostics.npz")
    np.testing.assert_array_equal(
        matrices["inverse_apriori_covariance_physical"][0, :], 0.0
    )
    prior_metadata = pd.read_csv(matrix_directory / "parameter_prior_metadata.csv")
    assert not bool(prior_metadata.loc[0, "prior_constrained"])
    assert pd.isna(prior_metadata.loc[0, "prior_sigma"])

    class Dynamics:
        def __init__(self, history):
            self.state_history_float = history

    class Simulation:
        def __init__(self, history):
            self.dynamics_results = Dynamics(history)

    first = np.arange(6.0)
    second = first + 1.0
    histories = [
        {0.: first - 1., 10.: first, 20.: first + 1.},
        {0.: second - 1., 10.: second, 20.: second + 1.},
    ]

    class IterationOutput:
        residual_history = np.array([[2e-3, 1e-3], [-2e-3, -1e-3]])
        simulation_results_per_iteration = [Simulation(item) for item in histories]
        best_iteration = 1

    rows = [{
        "index": index, "name": name,
        "unit": "m" if index < 3 else "m/s",
        "subarc_start_tdb": np.nan, "prior_sigma": None,
        "prior_constrained": False, "prior_information_diagonal": 0.0,
    } for index, name in enumerate(("x", "y", "z", "vx", "vy", "vz"))]
    rows.append({
        "index": 6, "name": "empirical_N_constant", "unit": "m/s^2",
        "subarc_start_tdb": 0., "prior_sigma": 2.0,
        "prior_constrained": True, "prior_information_diagonal": 0.25,
    })
    parameter_history = np.column_stack([
        np.r_[first, 0.], np.r_[second, 2.], np.r_[second + .5, 3.]
    ])

    def fake_orbit(history, start, end, step):
        return pd.DataFrame({
            "t": [0., 1., 2., 3., 4.], "R": np.arange(5.0),
            "T": np.zeros(5), "N": np.ones(5), "dx": np.zeros(5),
            "dy": np.zeros(5), "dz": np.zeros(5),
        })

    monkeypatch.setattr(campaign, "orbit_comparison", fake_orbit)
    selected = fake_orbit(histories[1], 0., 4., 1.)
    result = campaign.save_iteration_orbit_diagnostics(
        IterationOutput(), parameter_history, rows, 10., 0., 4., 1.,
        1., 3., selected, tmp_path / "iterations",
    )
    assert result[0]["unconstrained_parameter_count"] == 6
    assert result[0]["constrained_parameter_count"] == 1
    assert result[0]["parameter_update_to_next_normalized_l2"] == pytest.approx(1.0)
    updates = pd.read_csv(tmp_path / "iterations" / "iteration_parameter_updates.csv")
    state = updates[updates.parameter_index < 6]
    assert not state.prior_constrained.any()
    assert state.update_over_prior_sigma.isna().all()
    assert np.isfinite(state["update"]).all()


SHADOW_CASES = [
    ("078", "018", {"sun_radiation_shadowing_pixels"},
     {"sun_radiation_shadowing_pixels": 20}),
    ("079", "018", {"aerodynamic_shadowing_pixels"},
     {"aerodynamic_shadowing_pixels": 20}),
    ("080", "019", {"sun_radiation_shadowing_pixels"},
     {"sun_radiation_shadowing_pixels": 20}),
    ("081", "019", {"aerodynamic_shadowing_pixels"},
     {"aerodynamic_shadowing_pixels": 20}),
    ("082", "078", {"mars_radiation_target"},
     {"mars_radiation_target": "panelled"}),
    ("083", "082", {"mars_radiation_shadowing_pixels"},
     {"mars_radiation_shadowing_pixels": 20}),
    ("084", "080", {"mars_radiation_target"},
     {"mars_radiation_target": "panelled"}),
    ("085", "084", {"mars_radiation_shadowing_pixels"},
     {"mars_radiation_shadowing_pixels": 20}),
]


@pytest.mark.parametrize("case_id,control_id,changed,expected", SHADOW_CASES)
def test_final_shadow_configs_have_only_declared_source_specific_change(
        case_id, control_id, changed, expected):
    """Every final comparison has a named control and independent 0/20 switches."""
    control_path = (
        campaign.DEFAULT_ROOT / f"case_{control_id}" / "settings.json"
        if int(control_id) < 78 else
        campaign.DEFAULT_ROOT / "planned" / f"case_{control_id}.json"
    )
    control = json.loads(json.dumps(asdict(campaign.load_case(control_path))))
    candidate_case = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    candidate = json.loads(json.dumps(asdict(candidate_case)))
    actual = {
        key for key in control
        if key != "description" and control[key] != candidate[key]
    }
    assert actual == changed
    for key, value in expected.items():
        assert candidate[key] == value
    environment = candidate_case.environment()
    assert environment["MRO_SELF_SHADOWING_PIXELS"] == "0"
    assert environment["MRO_RADIATION_SELF_SHADOWING_PIXELS"] == "0"
    assert environment["MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS"] == str(
        candidate_case.sun_radiation_shadowing_pixels
    )
    assert environment["MRO_AERODYNAMIC_SELF_SHADOWING_PIXELS"] == str(
        candidate_case.aerodynamic_shadowing_pixels
    )
    assert environment["MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS"] == str(
        candidate_case.mars_radiation_shadowing_pixels
    )


@pytest.mark.parametrize("case_id,control_id", [
    ("078", "018"), ("079", "018"), ("080", "019"), ("081", "019"),
    ("082", "018"), ("083", "018"), ("084", "019"), ("085", "019"),
])
def test_final_shadow_native_model_parameter_and_prior_support(
        case_id, control_id, environments, kernels, monkeypatch):
    """Native shadow models retain every control parameter and prior index."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(candidate)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.),
        Time(378681900.), Time(378682100.))
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    total = 0
    for arc in range(7):
        original = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())["empirical_arc_start_times"]
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        persisted = pd.read_csv(
            campaign.DEFAULT_ROOT / f"case_{control_id}" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        columns = [
            "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
            "parameter_type",
        ]
        pd.testing.assert_frame_equal(
            table[columns].reset_index(drop=True),
            persisted[columns].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior), persisted.prior_sigma.to_numpy() ** -2
        )
        total += parameters.parameter_set_size
    assert total == (1282 if control_id == "018" else 864)


@pytest.mark.parametrize("case_id,changed,expected", [
    ("052", {"lift"}, {"lift": False}),
    ("053", set(), {}),
    ("054", {"reintegrate_variational"}, {"reintegrate_variational": True}),
    ("055", {"reduced_solar_arrays"}, {"reduced_solar_arrays": True}),
    ("033", {"mars_radiation_target"}, {"mars_radiation_target": "panelled"}),
    ("056", {"initial_position_offset_m"},
     {"initial_position_offset_m": [2.0, 2.0, 2.0]}),
    ("057", {"initial_position_offset_m"},
     {"initial_position_offset_m": [-2.0, -2.0, -2.0]}),
])
def test_final_case044_validations_have_only_declared_differences(
        case_id, changed, expected):
    """Approved final checks are exact case044 derivatives."""
    control = json.loads(json.dumps(asdict(campaign.load_case(
        campaign.DEFAULT_ROOT / "case_044" / "settings.json"
    ))))
    candidate = json.loads(json.dumps(asdict(campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    ))))
    actual = {
        key for key in control
        if key != "description" and control[key] != candidate[key]
    }
    assert actual == changed
    for key, value in expected.items():
        assert candidate[key] == value
    queue = json.loads((campaign.DEFAULT_ROOT / "RUN_QUEUE.json").read_text())
    job = next(item for item in queue["jobs"] if item["case_id"] == case_id)
    validated, reference, _, _ = campaign_queue.validate_job(
        campaign.DEFAULT_ROOT, job
    )


def test_case033_panelled_mars_keeps_case044_parameters_and_sun_model(
        environments, kernels, monkeypatch):
    """The approved slow case changes only the native Mars target model."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_044" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_033.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"mars_radiation_target"}
    environment = candidate.environment()
    assert environment["MRO_MARS_RADIATION_TARGET"] == "panelled"
    assert environment["MRO_REDUCED_SOLAR_ARRAY_MACROMODEL"] == "0"
    assert environment["MRO_RADIATION_SELF_SHADOWING_PIXELS"] == "0"
    assert environment["MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS"] == "0"
    assert environment["MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS"] == "0"
    assert candidate.sun_scale == control.sun_scale == "global"

    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    # This constructs the actual native panelled target and selects the native
    # paneled-target Mars acceleration; a missing/incompatible model fails here.
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    total = 0
    for arc in range(7):
        # Arc-wise aerodynamic coefficient objects are intentionally mutable
        # and reject a second estimated time history on the same body. Use a
        # fresh native environment per arc, matching real worker isolation.
        bodies = kernels.create_environment(
            Time(378671000.), Time(378693000.))[0]
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        original = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())["empirical_arc_start_times"]
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        persisted = pd.read_csv(
            campaign.DEFAULT_ROOT / "case_044" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        columns = [
            "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
            "parameter_type",
        ]
        pd.testing.assert_frame_equal(
            table[columns].reset_index(drop=True),
            persisted[columns].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
        total += parameters.parameter_set_size
    assert total == 871


def test_case019_fixed_sun_is_exact_case044_minus_sun_parameter(
        environments, kernels, monkeypatch):
    """Proposed economy test fixes only the seven nominal Sun scales."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    control = campaign.load_case(
        campaign.DEFAULT_ROOT / "case_044" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / "case_019.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    assert {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    } == {"sun_scale"}
    assert control.sun_scale == "global" and candidate.sun_scale == "fixed"

    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(candidate)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.),
        Time(378681900.), Time(378682100.))
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    total = 0
    for arc in range(7):
        original = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())["empirical_arc_start_times"]
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        persisted = pd.read_csv(
            campaign.DEFAULT_ROOT / "case_044" / "arcs"
            / f"arc_{arc:02d}" / "parameters.csv",
            float_precision="round_trip",
        )
        expected = persisted.loc[persisted.name != "sun_scale"].copy()
        expected["index"] = np.arange(len(expected))
        columns = [
            "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
            "parameter_type",
        ]
        pd.testing.assert_frame_equal(
            table[columns].reset_index(drop=True),
            expected[columns].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
        assert "sun_scale" not in set(table.name)
        total += parameters.parameter_set_size
    assert total == 864


@pytest.mark.parametrize("case_id,control_id,changed,expected_total", [
    ("010", "044", {"empirical_components", "empirical_shapes"}, 49),
    ("012", "011", {"lift_scale"}, 734),
    ("014", "011", {"empirical_shapes"}, 460),
    ("015", "011", {"empirical_components", "empirical_shapes"}, 186),
    ("018", "044", {"empirical_components"}, 1282),
    ("022", "044", {"empirical_sigma_m_s2", "constant_empirical_sigma_m_s2"}, 871),
    ("023", "044", {"empirical_sigma_m_s2", "constant_empirical_sigma_m_s2"}, 871),
    ("024", "011", {"drag_scale_sigma"}, 597),
    ("025", "011", {"drag_scale_sigma"}, 597),
    ("034", "048", {"mars_radiation_target"}, 871),
    ("040", "011", {"mcd_high_resolution"}, 597),
])
def test_remaining_defined_case_native_all_arc_parameter_prior_support(
        case_id, control_id, changed, expected_total,
        environments, kernels, monkeypatch):
    """Rebased defined jobs construct exact native parameter/prior maps."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup

    control = campaign.load_case(
        campaign.DEFAULT_ROOT / f"case_{control_id}" / "settings.json"
    )
    candidate = campaign.load_case(
        campaign.DEFAULT_ROOT / "planned" / f"case_{case_id}.json"
    )
    control_fields = json.loads(json.dumps(asdict(control)))
    candidate_fields = json.loads(json.dumps(asdict(candidate)))
    actual = {
        key for key in control_fields
        if key != "description" and control_fields[key] != candidate_fields[key]
    }
    assert actual == changed
    if case_id in {"024", "025"}:
        assert candidate.scale_sigma == 0.2
        assert candidate.drag_scale_sigma == (0.5 if case_id == "024" else 5.0)

    for key, value in candidate.environment().items():
        monkeypatch.setenv(key, value)
    validation = campaign.DEFAULT_ROOT / "subarc_diagnostic" / "validation"
    total = 0
    for arc in range(7):
        # Arc-wise aerodynamic coefficient objects reject a second estimated
        # time history on the same mutable body. Real arcs use isolated worker
        # processes, so reproduce that contract with a fresh body per arc.
        bodies = kernels.create_environment(
            Time(378671000.), Time(378693000.))[0]
        settings, _ = kernels.create_propagator_settings(
            bodies, "MRO", "Mars", Time(378682000.),
            Time(378681900.), Time(378682100.))
        original = json.loads((
            validation / f"case_001_arc_{arc:02d}" / "setup.json"
        ).read_text())["empirical_arc_start_times"]
        plan = campaign.ParameterPlan(candidate, arc)
        parameters = parameters_setup.create_parameter_set(
            plan.settings(settings, bodies, original), bodies, settings
        )
        inverse_prior = plan.priors(parameters)
        table = pd.DataFrame(plan.rows)
        assert len(table) == parameters.parameter_set_size
        assert table["index"].tolist() == list(range(len(table)))
        assert np.isfinite(table.prior_sigma).all()
        assert (table.prior_sigma > 0.0).all()
        np.testing.assert_array_equal(
            np.diag(inverse_prior), table.prior_sigma.to_numpy() ** -2
        )
        if candidate.drag_scale == "arcwise" or candidate.lift_scale == "arcwise":
            assert "empirical_T_constant" not in set(table.name)
            assert "empirical_T_sine" not in set(table.name)
            assert "empirical_T_cosine" not in set(table.name)
        if case_id in {"024", "025"}:
            expected_drag_sigma = 0.5 if case_id == "024" else 5.0
            assert set(table.loc[table.name == "drag_scale", "prior_sigma"]) == {
                expected_drag_sigma
            }
            assert set(table.loc[table.name == "sun_scale", "prior_sigma"]) == {0.2}
        total += parameters.parameter_set_size
    assert total == expected_total


def test_orbit_span_metrics_preserve_full_grid_split():
    orbit = pd.DataFrame({
        "t": [0., 1., 2., 3., 4.], "R": [1., 2., 3., 4., 5.],
        "T": np.zeros(5), "N": np.zeros(5),
    })
    summary, bracketed, edge = campaign.orbit_span_metrics(orbit, 1., 3.)
    assert bracketed.t.tolist() == [1., 2., 3.]
    assert edge.t.tolist() == [0., 4.]
    assert len(bracketed) + len(edge) == len(orbit)
    assert summary["bracketed_R_rms_m"] == pytest.approx(np.sqrt(29 / 3))
    assert summary["outside_edge_R_rms_m"] == pytest.approx(np.sqrt(13))


def test_iteration_orbit_diagnostics_align_state_parameter_and_best_indices(
        tmp_path, monkeypatch):
    class Dynamics:
        def __init__(self, history):
            self.state_history_float = history

    class Simulation:
        def __init__(self, history):
            self.dynamics_results = Dynamics(history)

    estimation_epoch = 10.0
    first_state = np.arange(6.0)
    second_state = first_state + 1.0
    histories = [
        {0.0: first_state - 1.0, estimation_epoch: first_state, 20.0: first_state + 1.0},
        {0.0: second_state - 1.0, estimation_epoch: second_state, 20.0: second_state + 1.0},
    ]

    class Output:
        residual_history = np.array([[2e-3, 1e-3], [-2e-3, -1e-3]])
        simulation_results_per_iteration = [Simulation(history) for history in histories]
        best_iteration = 1

    parameter_history = np.column_stack([first_state, second_state, second_state + .5])
    parameter_rows = [{
        "index": index,
        "name": name,
        "unit": "m" if index < 3 else "m/s",
        "subarc_start_tdb": np.nan,
        "prior_sigma": 10.0,
    } for index, name in enumerate(("x", "y", "z", "vx", "vy", "vz"))]

    def fake_orbit(history, start, end, step):
        offset = history[estimation_epoch][0]
        return pd.DataFrame({
            "t": [0., 1., 2., 3., 4.],
            "R": np.arange(5.0) + offset,
            "T": np.zeros(5),
            "N": np.ones(5),
            "dx": np.zeros(5),
            "dy": np.zeros(5),
            "dz": np.zeros(5),
        })

    monkeypatch.setattr(campaign, "orbit_comparison", fake_orbit)
    selected = fake_orbit(histories[1], 0., 4., 1.)
    rows = campaign.save_iteration_orbit_diagnostics(
        Output(), parameter_history, parameter_rows, estimation_epoch,
        0., 4., 1., 1., 3., selected, tmp_path,
    )
    assert [row["iteration"] for row in rows] == [0, 1]
    assert [row["is_best_iteration"] for row in rows] == [False, True]
    assert rows[0]["update_target_was_propagated_and_evaluated"]
    assert not rows[1]["update_target_was_propagated_and_evaluated"]
    assert rows[1]["bracketed_orbit_samples"] == 3
    assert rows[1]["outside_edge_orbit_samples"] == 2
    metadata = json.loads((tmp_path / "iteration_orbit_metrics.json").read_text())
    assert metadata["best_iteration"] == 1
    assert all(item["maximum_abs_state_index_error"] == 0.0
               for item in metadata["state_index_checks"])
    saved = np.load(tmp_path / "iteration_orbits.npz")
    assert saved["values"].shape == (2, 5, 6)
    assert saved["bracketed_mask"].tolist() == [False, True, True, True, False]
    updates = pd.read_csv(tmp_path / "iteration_parameter_updates.csv")
    assert len(updates) == 12
    assert not updates[updates.source_estimation_iteration == 1][
        "target_was_propagated_and_evaluated"
    ].any()

    mismatched = parameter_history.copy()
    mismatched[0, 1] += 1.0
    with pytest.raises(AssertionError):
        campaign.save_iteration_orbit_diagnostics(
            Output(), mismatched, parameter_rows, estimation_epoch,
            0., 4., 1., 1., 3., selected, tmp_path / "mismatch",
        )


def test_aggregate_iteration_orbit_diagnostics_pools_fixed_spans(tmp_path):
    for arc in range(7):
        directory = tmp_path / "arcs" / f"arc_{arc:02d}"
        directory.mkdir(parents=True)
        epochs = np.array([arc * 10., arc * 10. + 1.])
        values = np.zeros((2, 2, 6))
        values[0, :, 0] = 2.0
        values[1, :, 0] = 1.0
        np.savez_compressed(
            directory / "iteration_orbits.npz",
            epochs=epochs,
            values=values,
            bracketed_mask=np.array([False, True]),
            best_iteration=1,
        )
        np.save(directory / "residual_history.npy", np.array([[2e-3, 1e-3]]))
        campaign.write_json(directory / "iteration_objective_history.json", {
            "apply_apriori_parameter_deviation": True,
            "native_best_matches_reconstructed_total_cost": True,
            "per_iteration": [
                {
                    "finite": True,
                    "data_cost_half_rWr": 2e-6,
                    "absolute_prior_cost_half_delta_Pinv_delta": 0.0,
                    "objective_prior_cost": 0.0,
                    "total_cost": 2e-6,
                },
                {
                    "finite": True,
                    "data_cost_half_rWr": .5e-6,
                    "absolute_prior_cost_half_delta_Pinv_delta": .1e-6,
                    "objective_prior_cost": .1e-6,
                    "total_cost": .6e-6,
                },
            ],
        })
        pd.DataFrame({
            "source_estimation_iteration": [0, 1],
            "update_over_prior_sigma": [2.0, .5],
        }).to_csv(directory / "iteration_parameter_updates.csv", index=False)
    result = campaign.aggregate_iteration_orbit_diagnostics(tmp_path)
    assert result["iteration_count"] == 2
    assert result["all_arcs_share_best_iteration"]
    assert result["per_iteration"][0]["full_position_rms_m"] == 2.0
    assert result["per_iteration"][1]["full_position_rms_m"] == 1.0
    assert result["per_iteration"][1]["residual_rms_mhz"] == 1.0
    assert not result["per_iteration"][1][
        "update_target_was_propagated_and_evaluated"
    ]
    assert (tmp_path / "iteration_orbit_metrics.csv").is_file()
    assert (tmp_path / "iteration_orbits.npz").is_file()


def test_direct_spice_reference_and_rtn(kernels):
    """Independent SPICE units, interpolation and RTN rotation recover a known 3-m R offset."""
    import spiceypy
    history = {}
    for epoch in np.arange(378680000., 378684001., 10):
        state = np.asarray(spiceypy.spkezr("-74", epoch, "J2000", "NONE", "499")[0]) * 1000
        state[:3] += 3 * state[:3] / np.linalg.norm(state[:3])
        history[epoch] = state
    frame = campaign.orbit_comparison(history, 378681000., 378683000., 60.)
    np.testing.assert_allclose(frame.R, 3., atol=2e-8)
    np.testing.assert_allclose(frame[["T", "N"]], 0, atol=2e-8)
    with pytest.raises(ValueError, match="boundary"):
        campaign.orbit_comparison(history, 378680000., 378683000., 60.)


def test_pdf_and_global_metrics(tmp_path):
    """Synthetic per-arc results exercise aggregation/PDF output, without claiming a real fit."""
    campaign.write_json(tmp_path / "settings.json", asdict(campaign.cases()["001"]))
    for arc in range(7):
        directory = tmp_path / "arcs" / f"arc_{arc:02d}"
        directory.mkdir(parents=True)
        condition = "6.0e15" if arc == 6 else "1.0e10"
        (directory / "fit.log").write_text(
            f"Warning when performing least squares, condition number is {condition}\n")
        times = np.array([0., 60.]) + arc * 1000
        residuals = pd.DataFrame(dict(time=times, spice=[.001, -.001], prefit=[1., -1.],
                                     postfit=[.002, -.002], arc_index=arc))
        orbit = pd.DataFrame(dict(t=times, R=[1., -1.], T=[2., -2.], N=[0., 0.], arc_index=arc))
        parameters = pd.DataFrame(dict(name=["drag_scale"], unit=["1"], value=[1.1], delta=[.1],
                                       plot_start_tdb=[times[0]], plot_end_tdb=[times[-1]], arc_index=arc))
        for name, data in (("residuals", residuals), ("orbit", orbit), ("parameters", parameters)):
            data.to_csv(directory / f"{name}.csv", index=False)
        pd.DataFrame({"time": [times[0] + 30., times[-1]]}).to_csv(
            directory / "spice_residuals.csv", index=False)
        summary = campaign.metrics(residuals, orbit)
        summary["parameter_count"] = 1
        summary["best_iteration"] = 0
        campaign.write_json(directory / "summary.json", summary)
    summary = campaign.aggregate(tmp_path)
    assert summary["observations"] == 14 and summary["parameters_total"] == 7
    assert summary["position_rms_m"] == pytest.approx(np.sqrt(5))
    assert summary["mean_arc_position_rms_m"] == pytest.approx(np.sqrt(5))
    assert summary["median_arc_position_rms_m"] == pytest.approx(np.sqrt(5))
    assert not summary["condition_number_all_within_reference_ceiling"]
    assert (tmp_path / "results.pdf").stat().st_size > 1000


def test_primary_retained_tag_mask_is_inclusive_and_keeps_internal_gaps():
    orbit = pd.DataFrame({
        "t": [0., 60., 120., 180., 240.],
        "R": [100., 1., 2., 3., 100.],
        "T": np.zeros(5), "N": np.zeros(5),
    })
    # There is no observation at 120 s, but internal gaps are intentionally kept.
    retained = pd.DataFrame({"time": [60., 180.]})
    mask, metadata = primary_mask.retained_tag_mask(orbit, retained)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])
    assert metadata["inclusive_bounds"]
    assert metadata["primary_grid_samples"] == 3
    assert primary_mask.orbit_metrics(orbit[mask])["R_rms_m"] == pytest.approx(
        np.sqrt((1. + 4. + 9.) / 3.)
    )


def test_primary_pooled_rms_differs_from_equal_arc_average_and_promotes_iterations():
    first = pd.DataFrame({"R": [1.], "T": [0.], "N": [0.]})
    second = pd.DataFrame({"R": [3., 3., 3.], "T": [0.] * 3, "N": [0.] * 3})
    pooled = primary_mask.orbit_metrics(pd.concat([first, second], ignore_index=True))
    equal_arc_mean = np.mean([
        primary_mask.orbit_metrics(first)["position_rms_m"],
        primary_mask.orbit_metrics(second)["position_rms_m"],
    ])
    assert pooled["position_rms_m"] == pytest.approx(np.sqrt(7.))
    assert equal_arc_mean == pytest.approx(2.)
    assert pooled["position_rms_m"] != pytest.approx(equal_arc_mean)

    promoted = primary_mask.promote_iteration_metrics({
        "per_iteration": [{
            "full_position_rms_m": 9.,
            "bracketed_position_rms_m": 2.,
            "outside_edge_position_rms_m": 20.,
        }]
    })
    row = promoted["per_iteration"][0]
    assert row["position_rms_m"] == 2.
    assert row["full_nominal_position_rms_m"] == 9.
    assert "full_position_rms_m" not in row
    assert promoted["primary_orbit_mask_version"] == primary_mask.MASK_VERSION


def test_primary_mask_backfill_preserves_raw_orbit_and_full_nominal_audit(
        tmp_path, monkeypatch):
    directory = tmp_path / "case_001"
    all_orbits = []
    all_residuals = []
    all_parameters = []
    per_arc = []
    for arc in range(7):
        arc_directory = directory / "arcs" / f"arc_{arc:02d}"
        arc_directory.mkdir(parents=True)
        times = np.arange(5, dtype=float) * 60. + arc * 1000.
        orbit = pd.DataFrame({
            "t": times, "R": [100., 1., 2., 3., 100.],
            "T": np.zeros(5), "N": np.zeros(5),
            "dx": [100., 1., 2., 3., 100.],
            "dy": np.zeros(5), "dz": np.zeros(5), "arc_index": arc,
        })
        residuals = pd.DataFrame({
            "time": [times[1], times[3]], "spice": [0., 0.],
            "prefit": [0., 0.], "postfit": [0., 0.], "arc_index": arc,
        })
        parameters = pd.DataFrame({
            "name": ["x"], "unit": ["m"], "value": [0.], "delta": [0.],
            "plot_start_tdb": [times[0]], "plot_end_tdb": [times[-1]],
            "arc_index": [arc],
        })
        orbit.to_csv(arc_directory / "orbit.csv", index=False)
        residuals[["time"]].to_csv(arc_directory / "spice_residuals.csv", index=False)
        residuals.to_csv(arc_directory / "residuals.csv", index=False)
        parameters.to_csv(arc_directory / "parameters.csv", index=False)
        summary = {"status": "complete", "arc_index": arc, **primary_mask.orbit_metrics(orbit)}
        campaign.write_json(arc_directory / "summary.json", summary)
        campaign.write_json(arc_directory / "iteration_orbit_metrics.json", {
            "per_iteration": [{
                "full_position_rms_m": 10., "bracketed_position_rms_m": 2.,
            }]
        })
        (arc_directory / "results.pdf").write_bytes(b"legacy-pdf")
        all_orbits.append(orbit)
        all_residuals.append(residuals)
        all_parameters.append(parameters)
        per_arc.append(summary)
    root_orbit = pd.concat(all_orbits, ignore_index=True)
    pd.concat(all_orbits, ignore_index=True).to_csv(directory / "orbit.csv", index=False)
    pd.concat(all_residuals, ignore_index=True).to_csv(directory / "residuals.csv", index=False)
    pd.concat(all_parameters, ignore_index=True).to_csv(directory / "parameters.csv", index=False)
    campaign.write_json(directory / "settings.json", asdict(campaign.cases()["001"]))
    root_summary = {
        "status": "complete", "residual_rms_mhz": 0., "parameters_total": 7,
        **primary_mask.orbit_metrics(root_orbit), "per_arc": per_arc,
    }
    campaign.write_json(directory / "summary.json", root_summary)
    campaign.write_json(directory / "status.json", {
        key: value for key, value in root_summary.items() if key != "per_arc"
    })
    campaign.write_json(directory / "iteration_orbit_metrics.json", {
        "per_iteration": [{
            "full_position_rms_m": 10., "bracketed_position_rms_m": 2.,
        }]
    })
    (directory / "results.pdf").write_bytes(b"legacy-pdf")
    raw_before = (directory / "orbit.csv").read_bytes()
    def fake_plot(directory, *args, **kwargs):
        primary_mask._backup_once(
            directory / "results.pdf", directory / "results_full_nominal.pdf"
        )
        (directory / "results.pdf").write_bytes(b"primary-pdf")

    monkeypatch.setattr(primary_mask, "_plot_results_atomic", fake_plot)

    result = primary_mask.backfill_case(
        tmp_path, "001", primary_mask._reference_cutoffs(tmp_path)
    )
    assert (directory / "orbit.csv").read_bytes() == raw_before
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["primary_orbit_mask_version"] == primary_mask.MASK_VERSION
    assert summary["position_rms_m"] == pytest.approx(np.sqrt(14. / 3.))
    assert summary["full_nominal_position_rms_m"] > 60.
    assert (directory / "full_nominal_summary.json").is_file()
    assert (directory / "summary_legacy_before_primary_mask.json").is_file()
    assert result["plots_regenerated"] == 8
    iteration = json.loads((directory / "iteration_orbit_metrics.json").read_text())
    assert iteration["per_iteration"][0]["position_rms_m"] == 2.
    assert iteration["per_iteration"][0]["full_nominal_position_rms_m"] == 10.
    inventory = primary_mask.write_coverage_inventory(tmp_path)
    assert inventory["state"] == "complete_for_all_currently_completed_cases"
    assert inventory["converted_case_ids"] == ["001"]
    assert inventory["per_case"][0]["primary_pdfs"] == 8
    assert inventory["per_case"][0]["full_nominal_pdfs"] == 8
    persisted = json.loads(
        (tmp_path / "PRIMARY_MASK_BACKFILL_STATUS.json").read_text()
    )
    assert persisted["completed_case_count"] == 1
    assert persisted["converted_case_count"] == 1


def test_mcd_diagnostics_use_departure_from_one():
    """A tiny drag scale is not mistaken for a tiny correction to the nominal atmosphere."""
    table = pd.DataFrame(dict(name=["drag_scale", "drag_scale", "empirical_T_constant"],
                              unit=["1", "1", "m/s^2"], value=[0.1, 1.1, -2e-8],
                              plot_start_tdb=[0., 1., 0.], plot_end_tdb=[1., 4., 4.]))
    rows = {row["name"]: row for row in campaign.parameter_diagnostics(table)}
    assert rows["drag_scale"]["rms_correction"] == pytest.approx(np.sqrt((.9 ** 2 + 3 * .1 ** 2) / 4))
    assert rows["empirical_T_constant"]["mean_absolute"] == 2e-8
    assert "empirical_T_sine" not in rows


def test_memory_retries_halve_workers_and_preserve_logs(tmp_path, monkeypatch):
    """Simulate a killed worker; successful arcs must not run again or lose their logs."""
    attempts = {}
    class Child:
        def __init__(self, command, stdout, **kwargs):
            arc = int(command[command.index("--worker") + 1])
            attempts[arc] = attempts.get(arc, 0) + 1
            self.code = -9 if arc == 0 and attempts[arc] == 1 else 0
            stdout.write(b"Warning when performing least squares, condition number is 1.0e10\n")
        def wait(self, **kwargs):
            return self.code
        def poll(self):
            return self.code
    monkeypatch.setattr(campaign.subprocess, "Popen", Child)
    assert campaign.run_workers(tmp_path, {}, None, 7) == 4
    assert attempts == {0: 2, **{arc: 1 for arc in range(1, 7)}}
    assert "condition number is 1.0e10" in (
        tmp_path / "failed_attempts/arc_00_attempt_01/fit.log").read_text()


def test_condition_parser_handles_chunks_exponents_and_nonfinite_values():
    """A partial exponent must not be accepted before the complete line arrives."""
    scanner = campaign.ConditionLogScanner()
    assert scanner.feed(b"condition number is 4.9e") == []
    records = scanner.feed(b"+15\ncondition number is inf\ncondition number is na")
    assert records[0]["value"] == 4.9e15 and records[0]["passed"]
    assert records[1]["value"] is None and not records[1]["passed"]
    records = scanner.feed(b"n", final=True)
    assert records[0]["raw"].lower() == "nan" and not records[0]["passed"]


def test_condition_limit_is_inclusive():
    scanner = campaign.ConditionLogScanner(5.0e15)
    records = scanner.feed(
        "condition number is 5e15\ncondition number is 5.000000000000001e15\n",
        final=True,
    )
    assert records[0]["passed"]
    assert not records[1]["passed"]


def test_excessive_condition_is_recorded_without_aborting_siblings(tmp_path, monkeypatch):
    """A high diagnostic does not terminate otherwise successful workers."""
    children = []

    class Child:
        def __init__(self, command, stdout, **kwargs):
            self.arc = int(command[command.index("--worker") + 1])
            self.code = 0
            self.terminated = False
            value = "6e15" if self.arc == 2 else "1e10"
            stdout.write(
                f"Warning when performing least squares, condition number is {value}\n".encode()
            )
            children.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated = True
            self.code = -15

        def wait(self, **kwargs):
            return self.code

        def kill(self):
            self.code = -9

    monkeypatch.setattr(campaign.subprocess, "Popen", Child)
    assert campaign.run_workers(tmp_path, {}, None, 7) == 7
    assert len(children) == 7 and not any(child.terminated for child in children)
    report = json.loads((tmp_path / "condition_numbers.json").read_text())
    assert report["violating_arcs"] == [2]
    assert not report["all_within_reference_ceiling"]


def test_final_exit_race_is_recorded(tmp_path):
    """The mandatory final read catches output written during the last poll."""
    log = tmp_path / "arcs" / "arc_00" / "fit.log"
    log.parent.mkdir(parents=True)
    log.write_text("")

    class Child:
        code = None

        def poll(self):
            if self.code is None:
                with log.open("a") as stream:
                    stream.write("condition number is 9e15\n")
                self.code = 0
            return self.code

        def terminate(self):
            self.code = -15

        def wait(self, **kwargs):
            return self.code

        def kill(self):
            self.code = -9

    codes, diagnostics = campaign.monitor_children(
        tmp_path, [(0, Child(), log)], poll_seconds=0)
    assert codes == [0]
    assert diagnostics[0]["maximum_finite"] == 9e15
    assert not diagnostics[0]["within_reference_ceiling"]


def test_condition_scan_flags_excessive_without_rejection(tmp_path):
    for arc in range(7):
        directory = tmp_path / "arcs" / f"arc_{arc:02d}"
        directory.mkdir(parents=True)
        value = "5.1e15" if arc == 6 else "1e10"
        (directory / "fit.log").write_text(f"condition number is {value}\n")
    summary = campaign.scan_condition_logs(tmp_path)
    assert summary["violating_arcs"] == [6]
    assert not summary["all_within_reference_ceiling"]


def test_missing_condition_report_is_flagged_without_rejection(tmp_path):
    for arc in range(7):
        directory = tmp_path / "arcs" / f"arc_{arc:02d}"
        directory.mkdir(parents=True)
        (directory / "fit.log").write_text(
            "no native report\n" if arc == 4 else "condition number is 1e10\n")
    summary = campaign.scan_condition_logs(tmp_path)
    assert summary["missing_arcs"] == [4]
    assert not summary["diagnostic_complete"]


def test_failed_attempt_is_archived_whole(tmp_path):
    active = tmp_path / "case_001"
    active.mkdir()
    campaign.write_json(active / "settings.json", asdict(campaign.cases()["001"]))
    campaign.write_json(active / "status.json", {"status": "failed", "reason": "condition"})
    (active / "fit-evidence.txt").write_text("preserve me")
    target = campaign.archive_failed_run(tmp_path, "001", "condition limit")
    assert not active.exists()
    assert (target / "fit-evidence.txt").read_text() == "preserve me"
    assert json.loads((target / "failed_run.json").read_text())["reason_slug"] == "condition_limit"


def test_automatic_condition_prior_retries_are_disabled(tmp_path):
    with pytest.raises(ValueError, match="disabled"):
        campaign.launch_guarded(
            campaign.cases()["001"], "001", tmp_path,
            max_condition_prior_reductions=1,
        )


def test_guarded_baseline_priors_and_fixed_aero_scales():
    baseline = campaign.cases()["001"]
    assert baseline.observation_sigma_hz == 1.0
    assert baseline.position_sigma_m == 100.0
    assert baseline.velocity_sigma_m_s == 0.1
    assert baseline.scale_sigma == 0.2
    assert baseline.empirical_periods == 2.0
    assert baseline.drag_scale == baseline.lift_scale == "fixed"
    assert baseline.sun_scale == "global" and baseline.lift
    assert baseline.empirical_sigma_m_s2 == baseline.constant_empirical_sigma_m_s2 == 1e-6
    assert baseline.position_sigma_m ** -2 == pytest.approx(1e-4)
    assert baseline.velocity_sigma_m_s ** -2 == pytest.approx(100.0)
    assert baseline.scale_sigma ** -2 == pytest.approx(25.0)


def test_case003_changes_only_iteration_cap_from_case002():
    control = asdict(campaign.cases()["002"])
    convergence = asdict(campaign.cases()["003"])
    assert convergence.pop("iterations") == 8
    assert control.pop("iterations") == 5
    convergence.pop("description")
    control.pop("description")
    assert convergence == control


def test_anchored_cases_keep_edge_fix_and_change_only_prior_mode_and_iteration_cap():
    cases = campaign.cases()
    assert all(not cases[number].apply_apriori_parameter_deviation
               for number in ("001", "002", "003"))
    assert all(case.apply_apriori_parameter_deviation
               for number, case in cases.items() if int(number) >= 4)
    for anchored_number, legacy_number in (("004", "002"), ("005", "003")):
        anchored = asdict(cases[anchored_number])
        legacy = asdict(cases[legacy_number])
        assert anchored.pop("apply_apriori_parameter_deviation") is True
        assert legacy.pop("apply_apriori_parameter_deviation") is False
        anchored.pop("description")
        legacy.pop("description")
        assert anchored == legacy
        assert cases[anchored_number].empirical_edge_policy == "merge_case001_zero_edges"
        assert cases[anchored_number].environment()[
            "MRO_APPLY_APRIORI_PARAMETER_DEVIATION"
        ] == "1"
    assert all(case.iterations <= 5 for number, case in cases.items()
               if int(number) >= 6)


def test_case006_changes_only_integration_step_from_case004():
    control = asdict(campaign.cases()["004"])
    sensitivity = asdict(campaign.cases()["006"])
    assert control.pop("step_seconds") == 30.0
    assert sensitivity.pop("step_seconds") == 60.0
    control.pop("description")
    sensitivity.pop("description")
    assert sensitivity == control


def test_case007_changes_only_aerodynamic_law_from_case004():
    control = asdict(campaign.cases()["004"])
    storch = asdict(campaign.cases()["007"])
    assert control.pop("aerodynamic_model") == "variable_cross_section"
    assert storch.pop("aerodynamic_model") == "storch"
    assert not storch["reduced_solar_arrays"]
    control.pop("description")
    storch.pop("description")
    assert storch == control


def test_cancelled_and_adaptive_catalogue_ids_are_stable():
    cases = campaign.cases()
    assert "016" not in cases and "017" not in cases
    assert cases["015"].description == "Arc-wise drag, no empirical terms"
    assert cases["018"].description == "Fixed aerodynamic scales, R/T/N empirical terms"
    assert cases["020"].description.startswith("Approved adaptive")
    assert cases["021"].description.startswith("Approved adaptive")
    assert {
        "020", "021", "035", "038", "040", "043", "044", "045", "047",
        "075", "076", "077", "078", "079", "080", "081", "082", "083",
        "084", "085",
    } <= campaign.ADAPTIVE_CASE_IDS
    assert "040" not in campaign.DEFERRED_CASE_REASONS


def test_revised_mcd_high_resolution_placeholders_use_supported_scenario1_topologies():
    cases = campaign.cases()
    fixed, global_aero, arc_drag = (cases[number] for number in ("035", "038", "040"))
    assert all(case.mcd_scenario == 1 and case.mcd_high_resolution == 1
               for case in (fixed, global_aero, arc_drag))
    assert fixed.drag_scale == fixed.lift_scale == "fixed"
    assert fixed.empirical_components == "TN"
    assert global_aero.drag_scale == global_aero.lift_scale == "global"
    assert global_aero.empirical_components == "TN"
    assert arc_drag.drag_scale == "arcwise" and arc_drag.lift_scale == "fixed"
    assert "T" not in arc_drag.empirical_components
    assert all(cases[number].blocked_reason for number in ("036", "037", "039", "041", "042"))


@pytest.mark.parametrize("number,changed", [
    ("043", {"drag_scale", "lift_scale"}),
    ("044", {"empirical_sigma_m_s2", "constant_empirical_sigma_m_s2"}),
    ("045", {"lift"}),
    ("046", {"step_seconds"}),
    ("047", {"integrator"}),
])
def test_queue_catalogue_placeholders_have_only_declared_case004_differences(number, changed):
    control = asdict(campaign.cases()["004"])
    candidate = asdict(campaign.cases()[number])
    control.pop("description")
    candidate.pop("description")
    differences = {key for key in control if control[key] != candidate[key]}
    assert differences == changed


def test_rkf56_binding_and_fixed_step_constructor(monkeypatch):
    from tudatpy.dynamics import propagation_setup
    assert propagation_setup.integrator.rkf_56 == (
        propagation_setup.integrator.CoefficientSets.rkf_56
    )
    settings = propagation_setup.integrator.runge_kutta_fixed_step(
        30.0, propagation_setup.integrator.rkf_56
    )
    assert settings is not None
    assert campaign.cases()["047"].environment()[
        "MRO_INTEGRATOR_COEFFICIENT_SET"
    ] == "rkf56"


def _stub_queue(tmp_path):
    control = campaign.cases()["004"]
    control_dir = tmp_path / "case_004"
    control_dir.mkdir()
    campaign.write_json(control_dir / "settings.json", asdict(control))
    campaign.write_json(control_dir / "status.json", {"status": "complete"})
    campaign.write_json(control_dir / "summary.json", {
        "status": "complete", "observations": 5, "per_arc": [{}] * 7,
    })
    planned = tmp_path / "planned"
    planned.mkdir()
    candidates = {
        "100": replace(control, description="stub A", step_seconds=15.0),
        "101": replace(control, description="stub B", integrator="rkf56"),
    }
    for case_id, case in candidates.items():
        campaign.write_json(planned / f"case_{case_id}.json", asdict(case))
    jobs = [
        {
            "case_id": "100", "control_case_id": "004",
            "reference_case_id": "004", "config": "planned/case_100.json",
            "changed_fields": ["step_seconds"],
            "expected_values": {"step_seconds": 15.0}, "purpose": "stub A",
        },
        {
            "case_id": "101", "control_case_id": "004",
            "reference_case_id": "004", "config": "planned/case_101.json",
            "changed_fields": ["integrator"],
            "expected_values": {"integrator": "rkf56"}, "purpose": "stub B",
        },
    ]
    campaign.write_json(tmp_path / "RUN_QUEUE.json", {"version": 1, "jobs": jobs})
    return jobs


def _complete_stub_launch(calls):
    def launch(case, case_id, root, reference, workers):
        calls.append(case_id)
        directory = root / f"case_{case_id}"
        directory.mkdir()
        campaign.write_json(directory / "settings.json", asdict(case))
        campaign.write_json(directory / "status.json", {"status": "complete"})
        campaign.write_json(directory / "summary.json", {
            "status": "complete", "observations": 5,
            "per_arc": [{}] * 7, "residual_rms_mhz": 1.0,
            "R_rms_m": 1.0, "T_rms_m": 1.0, "N_rms_m": 1.0,
            "position_rms_m": 1.0, "worst_arc_position_rms_m": 1.0,
            "condition_number_max": 1.0, "final_workers": workers,
        })
    return launch


def test_queue_dispatches_stub_jobs_a_then_b_without_agent_turn(tmp_path):
    _stub_queue(tmp_path)
    calls = []
    campaign_queue.dispatch_queue(
        tmp_path, poll_seconds=0.001,
        launch_function=_complete_stub_launch(calls),
        wait_for_scientific_decision=False,
    )
    assert calls == ["100", "101"]
    status = json.loads((tmp_path / "QUEUE_STATUS.json").read_text())
    assert status["completed_case_ids"] == ["100", "101"]
    assert status["state"] == "needs_scientific_decision"


def test_queue_lock_excludes_second_dispatcher(tmp_path):
    _stub_queue(tmp_path)
    with campaign_queue.queue_lock(tmp_path):
        with pytest.raises(campaign_queue.QueueLockedError):
            campaign_queue.dispatch_queue(
                tmp_path, poll_seconds=0.001,
                launch_function=_complete_stub_launch([]),
                wait_for_scientific_decision=False,
            )


def test_queue_never_overwrites_completed_case_and_stops_on_failure(tmp_path):
    jobs = _stub_queue(tmp_path)
    calls = []
    good = _complete_stub_launch(calls)
    case = campaign.load_case(tmp_path / jobs[0]["config"])
    good(case, "100", tmp_path, tmp_path / "case_004", 7)
    marker = tmp_path / "case_100" / "preserve-me"
    marker.write_text("existing")

    def fail(case, case_id, root, reference, workers):
        calls.append(case_id)
        raise RuntimeError("stub failure")

    with pytest.raises(RuntimeError, match="stub failure"):
        campaign_queue.dispatch_queue(
            tmp_path, poll_seconds=0.001, launch_function=fail,
            wait_for_scientific_decision=False,
        )
    assert calls == ["100", "101"]
    assert marker.read_text() == "existing"
    status = json.loads((tmp_path / "QUEUE_STATUS.json").read_text())
    assert status["state"] == "paused_error"
    assert not (tmp_path / "case_101").exists()


def test_queue_accepts_nonfinite_condition_as_diagnostic_only(tmp_path):
    jobs = _stub_queue(tmp_path)
    calls = []
    launch = _complete_stub_launch(calls)
    case = campaign.load_case(tmp_path / jobs[0]["config"])
    launch(case, "100", tmp_path, tmp_path / "case_004", 7)
    summary_path = tmp_path / "case_100" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["condition_number_max"] = None
    campaign.write_json(summary_path, summary)
    validated = campaign_queue.validate_completed_result(tmp_path, jobs[0])
    assert validated["condition_number_max"] is None


def test_queue_carries_reduced_worker_cap_to_next_job(tmp_path):
    _stub_queue(tmp_path)
    calls = []

    def launch(case, case_id, root, reference, workers):
        calls.append((case_id, workers))
        directory = root / f"case_{case_id}"
        directory.mkdir()
        campaign.write_json(directory / "settings.json", asdict(case))
        campaign.write_json(directory / "status.json", {"status": "complete"})
        campaign.write_json(directory / "summary.json", {
            "status": "complete", "observations": 5,
            "per_arc": [{}] * 7, "residual_rms_mhz": 1.0,
            "R_rms_m": 1.0, "T_rms_m": 1.0, "N_rms_m": 1.0,
            "position_rms_m": 1.0, "worst_arc_position_rms_m": 1.0,
            "condition_number_max": 1.0,
            "final_workers": 3 if case_id == "100" else workers,
        })

    campaign_queue.dispatch_queue(
        tmp_path, poll_seconds=0.001, launch_function=launch,
        wait_for_scientific_decision=False,
    )
    assert calls == [("100", 7), ("101", 3)]
    assert json.loads((tmp_path / "QUEUE_STATUS.json").read_text())["worker_cap"] == 3


def test_queue_pauses_when_running_status_has_no_live_case_process(tmp_path):
    jobs = _stub_queue(tmp_path)
    case = campaign.load_case(tmp_path / jobs[0]["config"])
    directory = tmp_path / "case_100"
    directory.mkdir()
    campaign.write_json(directory / "settings.json", asdict(case))
    campaign.write_json(directory / "status.json", {"status": "running"})
    with pytest.raises(campaign_queue.QueueValidationError, match="no matching"):
        campaign_queue.dispatch_queue(
            tmp_path, poll_seconds=0.001,
            launch_function=_complete_stub_launch([]),
            wait_for_scientific_decision=False,
        )
    status = json.loads((tmp_path / "QUEUE_STATUS.json").read_text())
    assert status["state"] == "paused_error"


def test_queue_holds_at_pending_scientific_decision(tmp_path):
    jobs = _stub_queue(tmp_path)
    queue_path = tmp_path / "RUN_QUEUE.json"
    queue = json.loads(queue_path.read_text())
    queue["jobs"][0]["approval_state"] = "pending_after_control"
    campaign.write_json(queue_path, queue)
    calls = []
    campaign_queue.dispatch_queue(
        tmp_path, poll_seconds=0.001,
        launch_function=_complete_stub_launch(calls),
        wait_for_scientific_decision=False,
    )
    assert calls == []
    status = json.loads((tmp_path / "QUEUE_STATUS.json").read_text())
    assert status["state"] == "needs_scientific_decision"
    assert status["next_case_id"] == "100"


def test_register_display_tolerates_future_unlaunched_case_fields(tmp_path):
    planned = tmp_path / "planned"
    planned.mkdir()
    # Historical parent/restart logs share the case_* prefix but are not cases.
    (tmp_path / "case_009_parent_restart.log").write_text("retained log\n")
    raw = asdict(campaign.cases()["004"])
    raw["field_added_by_future_runner"] = "display-only tolerance"
    campaign.write_json(planned / "case_100.json", raw)
    campaign.write_json(planned / "case_040.json", asdict(campaign.cases()["040"]))
    campaign.update_register(tmp_path)
    assert "case_100" in (tmp_path / "CASES.md").read_text()
    register = pd.read_csv(tmp_path / "cases.csv").set_index("case")
    assert register.loc["case_040", "status"] == "planned"
    with pytest.raises(TypeError):
        campaign.load_case(planned / "case_100.json")


def test_postcompletion_reporting_typeerror_retains_results_and_queue_advances(
        tmp_path, monkeypatch):
    _stub_queue(tmp_path)
    calls = []

    def scientifically_complete_then_reporting_error(
            case, case_id, root, reference, workers):
        calls.append(case_id)
        directory = root / f"case_{case_id}"
        directory.mkdir()
        campaign.write_json(directory / "settings.json", asdict(case))
        campaign.write_json(directory / "status.json", {"status": "complete"})
        campaign.write_json(directory / "summary.json", {
            "status": "complete", "observations": 5,
            "per_arc": [{}] * 7, "residual_rms_mhz": 1.0,
            "R_rms_m": 1.0, "T_rms_m": 1.0, "N_rms_m": 1.0,
            "position_rms_m": 1.0, "worst_arc_position_rms_m": 1.0,
            "condition_number_max": 1.0, "final_workers": workers,
        })
        raise TypeError("simulated post-completion register incompatibility")

    monkeypatch.setattr(campaign, "launch", scientifically_complete_then_reporting_error)
    campaign_queue.dispatch_queue(
        tmp_path, poll_seconds=0.001,
        launch_function=campaign.launch_guarded,
        wait_for_scientific_decision=False,
    )
    assert calls == ["100", "101"]
    assert all((tmp_path / f"case_{case_id}" / "status.json").is_file()
               for case_id in calls)
    assert not (tmp_path / "failed_runs").exists()
    assert all((tmp_path / f"case_{case_id}" / "reporting_errors.json").is_file()
               for case_id in calls)
    status = json.loads((tmp_path / "QUEUE_STATUS.json").read_text())
    assert status["completed_case_ids"] == calls


def test_anchored_objective_uses_initial_parameter_reference_and_total_cost():
    class Output:
        residual_history = np.array([[1.0, 0.0]])
        parameter_history = np.array([[10.0, 12.0, 13.0]])
        best_iteration = 0

    rows = [{
        "index": 0, "name": "x", "unit": "m",
        "subarc_start_tdb": np.nan, "prior_sigma": 1.0,
    }]
    result = campaign.iteration_objective_history(
        Output(), rows, np.array([2.0]), np.array([[1.0]]), True
    )
    assert result["reference_parameter_history_column"] == 0
    assert result["per_iteration"][0]["data_cost_half_rWr"] == 1.0
    assert result["per_iteration"][0]["objective_prior_cost"] == 0.0
    assert result["per_iteration"][1]["data_cost_half_rWr"] == 0.0
    assert result["per_iteration"][1]["objective_prior_cost"] == 2.0
    assert result["reconstructed_total_cost_best_iteration"] == 0
    assert result["data_only_best_iteration"] == 1
    assert result["native_best_matches_reconstructed_total_cost"]
    assert result["native_best_differs_from_data_only_best"]
    assert result["per_iteration"][1]["prior_cost_by_parameter_group"][
        "initial_position"
    ] == 2.0


def test_installed_estimation_input_accepts_anchored_prior_flag():
    from tudatpy.estimation import estimation_analysis

    estimation_input = estimation_analysis.EstimationInput(
        None, apply_apriori_parameter_deviation=True
    )
    assert estimation_input is not None


def test_matrix_diagnostics_use_actual_prior_and_reconstruct_native_total(tmp_path):
    class Output:
        normalized_design_matrix = np.array([[0., 1.], [0., 2.]])
        normalization_terms = np.array([1., 3.])
        inverse_normalized_covariance = np.diag([4., 5.5])
        residual_history = np.array([[1., .5], [-1., -.5]])
        parameter_history = np.zeros((2, 3))
        correlations = np.eye(2)
        best_iteration = 1
        exception_during_inversion = False
        exception_during_propagation = False

    table = pd.DataFrame({
        "index": [0, 1],
        "name": ["zero_parameter", "observed_parameter"],
        "unit": ["1", "1"],
        "subarc_start_tdb": [np.nan, np.nan],
        "prior_sigma": [1., 1.],
    })
    observations = pd.DataFrame({
        "time": [10., 20.], "link_id": [3, 3], "link_ends": ["A - A", "A - A"],
        "msrType": ["doppler", "doppler"], "spice": [.01, -.01],
    })
    summary = campaign.save_estimation_diagnostics(
        Output(), observations, table, np.ones(2), np.diag([4., 4.5]), tmp_path)
    assert summary["zero_sensitivity_count"] == 1
    assert summary["prior_dominated_count"] == 1
    assert summary["reconstruction_matches_native"]
    columns = pd.read_csv(tmp_path / "conditioning_columns.csv")
    assert bool(columns.loc[0, "zero_sensitivity"])
    assert bool(columns.loc[0, "prior_dominates"])
    matrices = np.load(tmp_path / "normal_matrix_diagnostics.npz")
    np.testing.assert_allclose(matrices["prior_information_normalized"], np.diag([4., .5]))
    assert (tmp_path / "propagated_residual_iteration_00.csv").exists()
    assert (tmp_path / "propagated_residual_iteration_01.csv").exists()


def test_high_condition_matrix_uses_weak_modes_not_naive_correlations(tmp_path):
    class Output:
        normalized_design_matrix = np.array([[0., 1.], [0., 2.]])
        normalization_terms = np.array([1., 3.])
        inverse_normalized_covariance = np.diag([1e18, 5.5])
        residual_history = np.array([[1.], [-1.]])
        parameter_history = np.zeros((2, 2))
        best_iteration = 0
        exception_during_inversion = False
        exception_during_propagation = False

        @property
        def correlations(self):
            raise AssertionError("high-condition diagnostics must not request naive inverse correlations")

    table = pd.DataFrame({
        "index": [0, 1], "name": ["weak", "observed"], "unit": ["1", "1"],
        "subarc_start_tdb": [np.nan, np.nan], "prior_sigma": [1e-9, 1.],
    })
    observations = pd.DataFrame({
        "time": [10., 20.], "link_id": [3, 3], "link_ends": ["A - A", "A - A"],
        "msrType": ["doppler", "doppler"], "spice": [.01, -.01],
    })
    summary = campaign.save_estimation_diagnostics(
        Output(), observations, table, np.ones(2), np.diag([1e18, 4.5]), tmp_path)
    assert not summary["inverse_correlation_diagnostics_trustworthy"]
    assert summary["weak_singular_modes"]
    assert not (tmp_path / "posterior_correlations_trusted.npy").exists()


def test_nonfinite_later_residual_retains_earlier_columns_and_raw_matrix(tmp_path):
    class Output:
        normalized_design_matrix = np.array([[1.], [2.]])
        normalization_terms = np.array([2.])
        inverse_normalized_covariance = np.array([[5.25]])
        residual_history = np.array([[1., .5, np.nan], [-1., -.5, np.inf]])
        parameter_history = np.zeros((1, 4))
        correlations = np.eye(1)
        best_iteration = 0
        exception_during_inversion = False
        exception_during_propagation = False

    parameters = pd.DataFrame({
        "index": [0], "name": ["state"], "unit": ["m"],
        "subarc_start_tdb": [np.nan], "prior_sigma": [1.],
    })
    observations = pd.DataFrame({
        "time": [10., 20.], "link_id": [3, 3], "link_ends": ["A - A", "A - A"],
        "msrType": ["doppler", "doppler"], "spice": [.01, -.01],
    })
    with pytest.raises(ValueError, match=r"iterations \[2\]"):
        campaign.save_estimation_diagnostics(
            Output(), observations, parameters, np.ones(2), np.array([[1.]]), tmp_path)
    assert (tmp_path / "propagated_residual_iteration_00.csv").exists()
    assert (tmp_path / "propagated_residual_iteration_01.csv").exists()
    assert (tmp_path / "propagated_residual_iteration_02.csv").exists()
    history = np.load(tmp_path / "residual_history.npy")
    np.testing.assert_allclose(history[:, :2], Output.residual_history[:, :2])
    assert np.isnan(history[0, 2]) and np.isinf(history[1, 2])
    flags = json.loads((tmp_path / "residual_history_diagnostics.json").read_text())
    assert [item["finite"] for item in flags["per_iteration"]] == [True, True, False]
    raw = np.load(tmp_path / "normal_matrix_inputs_raw.npz")
    np.testing.assert_allclose(raw["inverse_normalized_covariance_native"], [[5.25]])


@pytest.mark.parametrize("number", ["004", "012", "034", "038"])
def test_short_variational_propagation(number, environments, kernels, monkeypatch):
    """Evaluate actual forces/partials for two minutes, not a multi-day fit."""
    from tudatpy.astro.time_representation import Time
    from tudatpy.dynamics import parameters_setup, simulator
    case = campaign.cases()[number]
    for key, value in case.environment().items():
        monkeypatch.setenv(key, value)
    bodies = environments(case)
    settings, _ = kernels.create_propagator_settings(
        bodies, "MRO", "Mars", Time(378682000.), Time(378681940.), Time(378682060.))
    plan = campaign.ParameterPlan(case, arc_index=0)
    parameters = parameters_setup.create_parameter_set(
        plan.settings(settings, bodies, native_test_empirical_boundaries(case)),
        bodies, settings)
    plan.priors(parameters)
    solver = simulator.create_variational_equations_solver(bodies, settings, parameters)
    result = solver.dynamics_simulator.propagation_results
    assert result.integration_completed_successfully
    states = np.asarray(list(result.state_history_float.values()))
    assert np.isfinite(states).all() and len(states) >= 5


def test_reduced_arrays_preserve_face_areas_and_normals():
    """Only thin edge panels may disappear; neither illuminated face loses material or area."""
    from mro_utils import macromodel_mro
    full = macromodel_mro(False).panel_settings_list
    reduced = macromodel_mro(True).panel_settings_list
    assert (len(full), len(reduced)) == (116, 88)
    for panels, n in ((full, 22), (reduced, 8)):
        array = panels[-n:]
        normal = np.array([p.panel_geometry.surface_normal for p in array])
        area = np.array([p.panel_geometry.area for p in array])
        for sign, material in ((1, "SA_front"), (-1, "SA_back")):
            selected = normal[:, 2] * sign > .99
            assert area[selected].sum() == pytest.approx(11.57400109947103, rel=1e-10)
            assert {p.panel_type_id for p, keep in zip(array, selected) if keep} == {material}
    # Non-array geometry is unchanged, and both rotations still come from the same SPICE frames.
    for original, simplified in zip(full[:-44], reduced[:-16]):
        assert original.panel_geometry.area == simplified.panel_geometry.area
        np.testing.assert_array_equal(original.panel_geometry.surface_normal, simplified.panel_geometry.surface_normal)
        assert original.panel_type_id == simplified.panel_type_id

    # The native panel settings do not expose triangle centroids.  Verify the
    # source meshes directly so the reduction cannot preserve area while moving
    # or shrinking the illuminated footprint.
    def face_geometry(path):
        namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
        root = ElementTree.parse(path).getroot()
        sources = root.findall(".//c:source", namespace)
        positions_source = next(
            source for source in sources if source.attrib["id"].endswith("positions")
        )
        normals_source = next(
            source for source in sources if source.attrib["id"].endswith("normals")
        )
        positions = np.fromstring(
            positions_source.find("c:float_array", namespace).text, sep=" "
        ).reshape(-1, 3) * 1.0e-3
        normals = np.fromstring(
            normals_source.find("c:float_array", namespace).text, sep=" "
        ).reshape(-1, 3)
        result = {}
        omitted_edge_area = 0.0
        for triangles in root.findall(".//c:triangles", namespace):
            indices = np.fromstring(
                triangles.find("c:p", namespace).text, sep=" ", dtype=int
            ).reshape(-1, 3, 2)
            face_areas, face_centroids, face_vertices = [], [], []
            for triangle in indices:
                vertices = positions[triangle[:, 0]]
                normal = normals[triangle[0, 1]]
                area = 0.5 * np.linalg.norm(
                    np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                )
                if abs(normal[2]) > 0.99:
                    face_areas.append(area)
                    face_centroids.append(vertices.mean(axis=0))
                    face_vertices.extend(vertices)
                else:
                    omitted_edge_area += area
            if face_areas:
                face_areas = np.asarray(face_areas)
                face_vertices = np.asarray(face_vertices)
                result[triangles.attrib["material"]] = {
                    "area": face_areas.sum(),
                    "centroid": np.average(
                        np.asarray(face_centroids), axis=0, weights=face_areas
                    ),
                    "minimum": face_vertices.min(axis=0),
                    "maximum": face_vertices.max(axis=0),
                }
        return result, omitted_edge_area

    mesh_directory = Path(__file__).resolve().parent / "mro_macromodel"
    full_faces, full_edge_area = face_geometry(mesh_directory / "MRO_sa.dae")
    reduced_faces, reduced_edge_area = face_geometry(
        mesh_directory / "MRO_sa_reduced.dae"
    )
    assert full_faces.keys() == reduced_faces.keys() == {
        "SA_front-material", "SA_back-material"
    }
    for material in full_faces:
        for field in ("area", "centroid", "minimum", "maximum"):
            np.testing.assert_allclose(
                reduced_faces[material][field], full_faces[material][field],
                rtol=0.0, atol=1.0e-14,
            )
    assert full_edge_area == pytest.approx(0.4218781331878225, abs=1.0e-14)
    assert reduced_edge_area == 0.0
