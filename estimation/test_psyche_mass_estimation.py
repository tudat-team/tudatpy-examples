"""Verify joint covariance bookkeeping and native dynamical coupling without downloads."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import psyche_mass_estimation as joint
from tudatpy.dynamics import simulator

od = joint.od


def test_sbdb_asymmetric_error_keeps_the_diameter_value_and_its_unit(monkeypatch):
    from astroquery.jplsbdb import SBDB

    class ParsedQuery:
        @property
        def diameter(self):
            raise ValueError("Astroquery left the diameter as a string")

    monkeypatch.setattr(od, "SBDBquery", lambda target: ParsedQuery())
    monkeypatch.setattr(SBDB, "query_async", lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {"phys_par": [{"name": "diameter", "value": "222", "units": "km", "sigma": "-1/+4"}]}))
    assert od.sbdb_photocenter_radius("16") == 111000.
    monkeypatch.setattr(SBDB, "query_async", lambda *args, **kwargs: SimpleNamespace(
        json=lambda: {"phys_par": []}))
    with pytest.raises(ValueError, match="supply its photocenter radius"):
        od.sbdb_photocenter_radius("67125")


def test_body_selection_preserves_correlated_weights_and_rejected_event_ids():
    obs = od.observations
    dataset = obs.ObservationDataset()
    for body in ("673", "16", "673"):
        link = obs.LinkDefinition({obs.transmitter: obs.LinkEndId(body, ""),
                                   obs.receiver: obs.LinkEndId("Gaia", "")})
        dataset.add_observation_set(obs.angular_position_type, link,
                                    np.zeros((2, 2)), [1., 2.], obs.receiver)
    covariance = np.diag([1., 2., 3., 4.]) + 0.25 * np.ones((4, 4))
    weights = np.linalg.inv(covariance)
    for set_id in range(3):
        dataset.set_weight_matrix_for_set(set_id, weights)
    q = obs.observation_query
    dataset.reject_observations((q.set_id == 2) & (q.time == 1.), "test rejection")
    selected = joint.dataset_for_body(dataset, "673")
    assert selected.get_observation_ids() == [0, 1, 4, 5]
    assert selected.get_observation_ids(q.active) == [0, 1, 5]
    components = dataset.get_scalar_components(ordering="estimation")
    rows = [components.index(component) for component in selected.get_scalar_components(ordering="estimation")]
    np.testing.assert_array_equal(selected.get_weight_matrix().toarray(),
                                  dataset.get_weight_matrix().toarray()[np.ix_(rows, rows)])
    output = SimpleNamespace(residual_history=np.arange(24).reshape(12, 2),
                             active_flags_per_iteration=np.ones((12, 2), dtype=bool))
    view = joint.output_for_body(output, dataset, selected)
    np.testing.assert_array_equal(view.residual_history, output.residual_history[rows])
    metadata = joint.scalar_observation_metadata(dataset, ["16", "673"])
    assert len(metadata["scalar_bodies"]) == output.residual_history.shape[0]
    for index, (event, component) in enumerate(metadata["scalar_components"]):
        assert metadata["scalar_bodies"][index] == ("16" if event in (2, 3) else "673")
        assert metadata["scalar_times"][index] == (1. if event % 2 == 0 else 2.)


@pytest.fixture
def coupled_setup(monkeypatch):
    # A short synthetic close encounter isolates the wiring of the existing force model.
    # Production retains all 118 perturbers; this structural test needs only Psyche.
    monkeypatch.setattr(od, "ASTEROID_PERTURBERS", [(16, "Psyche")])
    od.spice.load_standard_kernels()
    psyche = od.spice.get_body_cartesian_state_at_epoch("2000016", "Sun", "J2000", "NONE", 0.)
    target = psyche + np.array([2.e9, 1.e9, 3.e8, 0., 50., 10.])
    initial = np.concatenate((psyche, target))
    end = 30 * od.constants.JULIAN_DAY
    references = {body: {float(t): state.copy() for t in np.linspace(-end, 2*end, 12)}
                  for body, state in zip(("16", "673"), (psyche, target))}
    bodies = od.create_bodies(0., end, reference_state_history=references,
                              estimated_bodies=["16", "673"])
    accelerations = od.propagation_setup.create_acceleration_models(
        bodies, od.acceleration_settings(False, ["16", "673"]), ["16", "673"], ["Sun", "Sun"])

    def live_psyche_force():
        sun = bodies.get("Sun").state[:3]
        return np.concatenate((bodies.get("16").state[:3] - sun,
                               bodies.get("673").state[:3] - sun,
                               accelerations["673"]["16"][0].acceleration))

    propagator = od.propagation_setup.propagator.translational(
        central_bodies=["Sun", "Sun"], acceleration_models=accelerations,
        bodies_to_integrate=["16", "673"], initial_states=initial, initial_time=0.,
        integrator_settings=od.propagation_setup.integrator.runge_kutta_fixed_step(
            od.time_representation.Time(od.INTEGRATOR_STEP), od.propagation_setup.integrator.CoefficientSets.rkf_45),
        termination_settings=od.propagation_setup.propagator.time_termination(end),
        output_variables=[od.propagation_setup.dependent_variable.custom_dependent_variable(live_psyche_force, 9)])
    settings = od.parameters_setup.initial_states(propagator, bodies)
    settings.append(od.parameters_setup.gravitational_parameter("16"))
    parameters = od.parameters_setup.create_parameter_set(settings, bodies, propagator)
    solver = simulator.create_variational_equations_solver(bodies, propagator, parameters)
    return bodies, accelerations, propagator, parameters, solver, initial, end


def test_psyche_propagated_state_and_gm_drive_the_encounter(coupled_setup):
    bodies, accelerations, propagator, parameters, solver, initial, end = coupled_setup
    assert not bodies.does_body_exist("16 Psyche")
    assert parameters.parameter_set_size == 13
    ids = parameters.get_parameter_identifiers()
    assert [identifier[1][0] for identifier in ids[:2]] == ["16", "673"]
    assert parameters.indices_for_parameter_type(ids[0]) == [(0, 6)]
    assert parameters.indices_for_parameter_type(ids[1]) == [(6, 6)]
    assert parameters.indices_for_parameter_type(ids[2]) == [(12, 1)]
    result = solver.variational_propagation_results
    history = result.dynamics_results.state_history_float
    final = np.asarray(history[end])
    assert np.linalg.norm(final[:3] - initial[:3]) > 1.e8
    live = np.asarray(result.dynamics_results.dependent_variable_history[end])
    psyche, encounter = live[:3], live[3:6]
    # The environment's live state must be the integrated state, not the constant reference table.
    np.testing.assert_allclose(psyche, final[:3], rtol=0., atol=1.e-3)
    gm = parameters.parameter_vector[12]
    expected = gm * ((psyche-encounter)/np.linalg.norm(psyche-encounter)**3
                     - psyche/np.linalg.norm(psyche)**3)
    np.testing.assert_allclose(live[6:], expected, rtol=1.e-10)
    matrix = solver.state_transition_interface.full_state_transition_sensitivity_at_epoch(end)
    assert matrix.shape == (12, 13)
    assert np.linalg.norm(matrix[6:9, :3]) > 1.e-9
    assert np.linalg.norm(matrix[6:9, 12]) > 1.e-9

    nominal_parameters = parameters.parameter_vector.copy()
    changed = nominal_parameters.copy()
    delta_gm = 0.01 * gm
    changed[12] += delta_gm
    parameters.parameter_vector = changed
    assert bodies.get("16").gravity_field_model.gravitational_parameter == pytest.approx(1.01 * gm)
    solver.integrate_equations_of_motion_only(initial)
    gm_final = np.asarray(solver.variational_propagation_results.dynamics_results.state_history_float[end])
    np.testing.assert_allclose((gm_final[6:9]-final[6:9])/delta_gm, matrix[6:9, 12], rtol=0.02, atol=1.e-11)
    changed = nominal_parameters.copy()
    changed[0] += 1.e6
    parameters.parameter_vector = changed
    solver.integrate_equations_of_motion_only(changed[:12])
    state_final = np.asarray(solver.variational_propagation_results.dynamics_results.state_history_float[end])
    np.testing.assert_allclose((state_final[6:9]-final[6:9])/1.e6, matrix[6:9, 0], rtol=0.02, atol=1.e-9)


def test_optional_yarkovsky_parameters_are_retained_for_both_bodies(coupled_setup):
    bodies, _, propagator, _, _, _, _ = coupled_setup
    propagator.reset_and_recreate_acceleration_models(od.acceleration_settings(True, ["16", "673"]), bodies)
    settings = od.parameters_setup.initial_states(propagator, bodies)
    settings.append(od.parameters_setup.gravitational_parameter("16"))
    settings.extend(od.parameters_setup.yarkovsky_parameter(body, "Sun") for body in ("16", "673"))
    parameters = od.parameters_setup.create_parameter_set(settings, bodies, propagator)
    assert parameters.parameter_set_size == 15
    for body, index in (("16", 13), ("673", 14)):
        identifier = od.parameters_setup.yarkovsky_parameter(body, "Sun").parameter_identifier
        assert parameters.indices_for_parameter_type(identifier) == [(index, 1)]
    assert joint.parameter_labels(parameters)[12] == "16: gravitational parameter"
    assert joint.parameter_labels(parameters)[13] == "16: Yarkovsky A2"


def test_ten_body_native_variational_system_contains_all_states_and_mass(monkeypatch):
    monkeypatch.setattr(od, "ASTEROID_PERTURBERS", [(16, "Psyche")])
    od.spice.load_standard_kernels()
    psyche = od.spice.get_body_cartesian_state_at_epoch("2000016", "Sun", "J2000", "NONE", 0.)
    selected = ["16", "673"] + [str(100000 + i) for i in range(8)]
    states = [psyche] + [psyche + index * np.array([2.e9, 1.e9, 3.e8, 0., 50., 10.])
                         for index in range(1, len(selected))]
    # Supply enough epochs for Tudat's sensitivity interpolator, and create the
    # complete environment before propagating, exactly as production does.
    end = 10 * od.INTEGRATOR_STEP
    references = {body: {float(t): state.copy() for t in np.linspace(-end, 2*end, 12)}
                  for body, state in zip(selected, states)}
    bodies = od.create_bodies(0., end, reference_state_history=references, estimated_bodies=selected)
    accelerations = od.propagation_setup.create_acceleration_models(
        bodies, od.acceleration_settings(False, selected), selected, ["Sun"] * len(selected))
    propagator = od.propagation_setup.propagator.translational(
        central_bodies=["Sun"] * len(selected), acceleration_models=accelerations,
        bodies_to_integrate=selected, initial_states=np.concatenate(states), initial_time=0.,
        integrator_settings=od.propagation_setup.integrator.runge_kutta_fixed_step(
            od.time_representation.Time(od.INTEGRATOR_STEP), od.propagation_setup.integrator.CoefficientSets.rkf_45),
        termination_settings=od.propagation_setup.propagator.time_termination(end))
    settings = od.parameters_setup.initial_states(propagator, bodies)
    settings.append(od.parameters_setup.gravitational_parameter("16"))
    parameters = od.parameters_setup.create_parameter_set(settings, bodies, propagator)
    solver = simulator.create_variational_equations_solver(bodies, propagator, parameters)
    assert len(solver.variational_propagation_results.dynamics_results.state_history_float) == 11
    matrix = solver.state_transition_interface.full_state_transition_sensitivity_at_epoch(end)
    assert parameters.parameter_set_size == 61
    assert matrix.shape == (60, 61)
    assert np.all(np.isfinite(matrix))
    for index in range(1, len(selected)):
        assert np.linalg.norm(matrix[6*index:6*index+3, :3]) > 0
        assert np.linalg.norm(matrix[6*index:6*index+3, 60]) > 0


def test_multiple_gaia_targets_keep_independent_full_transit_blocks(coupled_setup):
    bodies = coupled_setup[0]
    bodies.create_empty_body("Gaia")
    data = {}
    for body in ("16", "673"):
        table = joint.pd.DataFrame({"number_mp": [int(body)] * 2, "epoch": [1., 2.],
            "transit_id": [42, 42], "ra": [0.1, 0.2], "dec": [0.2, 0.3],
            "ra_error_random": [1., 2.], "dec_error_random": [2., 3.],
            "ra_dec_correlation_random": [0.3, -0.1],
            "ra_error_systematic": [3., 3.], "dec_error_systematic": [4., 4.],
            "ra_dec_correlation_systematic": [0.5, 0.5]})
        data[body] = {"gaia": od.GaiaAstrometry(table)}
    combined = joint.merge_gaia(data)
    tracks, _ = combined.to_tracking_data()
    dataset = od.observations.create_observation_dataset_from_tracking_data(tracks, bodies)
    systematic = np.array([[9., 6.], [6., 16.]])
    random = np.zeros((4, 4))
    random[:2, :2] = [[1., 0.6], [0.6, 4.]]
    random[2:, 2:] = [[4., -0.6], [-0.6, 9.]]
    expected_weights = np.linalg.inv(random + np.tile(systematic, (2, 2)))
    for body in ("16", "673"):
        subset = joint.dataset_for_body(dataset, body)
        assert len(subset.get_scalar_components()) == 4
        np.testing.assert_allclose(subset.get_weight_matrix().toarray(), expected_weights, atol=1.e-14)
