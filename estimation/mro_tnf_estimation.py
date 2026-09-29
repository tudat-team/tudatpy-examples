# %%
import os
import re
from collections.abc import Callable, Sequence
from typing import Any
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import multiprocessing
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


from mro_utils import macromodel_mro

from tudatpy.data_input.environment_data import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.data_input.tracking_data import TrackingData, TrackingSupplementaryData
from tudatpy.data_input.tracking_data.tnf import OpenRampHandling, read_tnf_data
from tudatpy.dynamics import environment, environment_setup, parameters, propagation_setup, parameters_setup, propagation
from tudatpy.estimation import estimation_analysis, observable_models_setup, observations, observations_setup

from tudatpy.math import interpolators

import time as t


HERE = Path(__file__).resolve().parent
ESTIMATION_ARCS = (
    ("2012-01-01 03:18:01.965", "2012-01-04 01:58:15.132"),
    ("2012-01-04 02:25:09.706", "2012-01-07 02:55:23.122"),
    ("2012-01-07 03:23:14.407", "2012-01-10 02:03:27.113"),
    ("2012-01-10 02:22:44.539", "2012-01-13 02:52:58.104"),
    ("2012-01-13 03:15:38.112", "2012-01-16 02:05:51.095"),
    ("2012-01-16 02:22:44.352", "2012-01-19 02:52:57.085"),
    ("2012-01-19 03:17:17.831", "2012-01-22 01:57:31.076"),
)
StateHistory = dict[float, np.ndarray]
ParameterMetadata = list[dict[str, Any]]
ArcResult = dict[str, Any]


# =============================================================================================
# ================   LOAD AND PREPROCESS TRACKING DATA   ======================================
# =============================================================================================


def prepare_arc_inputs(
    arc_index: int, start_epoch: datetime, end_epoch: datetime
) -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str], str, str
]:
    """Select local kernels, media corrections, and TNF days overlapping an arc."""
    print("Selecting input files for the arc...")
    kernel_directory = HERE / "mro_kernels"
    lower = start_epoch - timedelta(days=1)
    upper = end_epoch + timedelta(days=1)

    orientation_files = []
    for prefix in ("sc", "hga", "sa"):
        selected = []
        for path in sorted(kernel_directory.glob(f"mro_{prefix}_psp_*.bc")):
            dates = re.search(r"(\d{6})_(\d{6})\.bc$", path.name)
            if dates is None:
                continue
            first, last = [
                datetime.strptime(value, "%y%m%d") for value in dates.groups()
            ]
            if first <= upper and last + timedelta(days=1) >= lower:
                selected.append(str(path))
        if not selected:
            raise FileNotFoundError(
                f"No {prefix} attitude kernels overlap arc {arc_index}."
            )
        orientation_files.extend(selected)

    media_files = []
    for suffix in ("tro", "ion"):
        selected = []
        for path in sorted(kernel_directory.glob(f"mromagr*.{suffix}")):
            dates = re.search(r"(\d{4}_\d{3})_(\d{4}_\d{3})", path.name)
            if dates is None:
                continue
            first, last = [
                datetime.strptime(value, "%Y_%j") for value in dates.groups()
            ]
            if first <= upper and last >= lower:
                selected.append(str(path))
        if not selected:
            raise FileNotFoundError(
                f"No {suffix} corrections overlap arc {arc_index}."
            )
        media_files.append(selected)
    tro_files, ion_files = media_files

    tnf_files = []
    for path in sorted(kernel_directory.glob("*.tnf")):
        stamp = re.search(r"mromagr(\d{4}_\d{3})", path.name)
        if stamp is None:
            continue
        day = datetime.strptime(stamp.group(1), "%Y_%j").date()
        if lower.date() <= day <= end_epoch.date():
            tnf_files.append(str(path))
    if not tnf_files:
        raise FileNotFoundError(f"No TNF data overlap arc {arc_index}.")

    clock_files = [str(kernel_directory / "mro_sclkscet_00112_65536.tsc")]
    trajectory_files = [str(kernel_directory / "mro_psp22.bsp")]
    frames_def_file = str(kernel_directory / "mro_v16.tf")
    structure_file = str(kernel_directory / "mro_struct_v10.bsp")
    required_files = (
        clock_files
        + orientation_files
        + tro_files
        + ion_files
        + tnf_files
        + trajectory_files
        + [frames_def_file, structure_file]
    )
    missing_files = [path for path in required_files if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required MRO input files: " + ", ".join(missing_files)
        )

    return (
        tnf_files,
        clock_files,
        orientation_files,
        tro_files,
        ion_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    )


def load_spice_kernels(
    clock_files: Sequence[str],
    orientation_files: Sequence[str],
    trajectory_files: Sequence[str],
    frames_def_file: str,
    structure_file: str,
) -> None:
    """Load the standard and MRO-specific SPICE kernels."""
    print("Loading SPICE kernels...")
    spice.load_standard_kernels()

    for orientation_file in orientation_files:
        spice.load_kernel(orientation_file)
    for clock_file in clock_files:
        spice.load_kernel(clock_file)
    spice.load_kernel(frames_def_file)
    for trajectory_file in trajectory_files:
        spice.load_kernel(trajectory_file)
    spice.load_kernel(structure_file)


def load_tracking_data(
    tnf_files: Sequence[str],
) -> tuple[list[TrackingData], list[TrackingSupplementaryData]]:
    """Load MRO Doppler observations and their supplementary data."""
    print("Loading TNF tracking data...")
    return read_tnf_data(
        tnf_files,
        ["doppler"],
        spacecraft_name="MRO",
        open_ramp_handling=OpenRampHandling.close_silently,
    )


# =============================================================================================
# ================   CREATE THE SIMULATION ENVIRONMENT   ======================================
# =============================================================================================


def create_environment(
    environment_start_time: time_representation.Time,
    environment_end_time: time_representation.Time,
) -> environment.SystemOfBodies:
    """Create the celestial-body and MRO environment for one estimation arc."""
    print("Setting up simulation environment...")
    bodies_to_create = [
        "Earth",
        "Sun",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Phobos",
        "Deimos",
    ]
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    start_epoch = environment_start_time.to_float()
    end_epoch = environment_end_time.to_float()
    hourly_interpolator = interpolators.interpolator_generation_settings(
        interpolators.cubic_spline_interpolation(),
        start_epoch,
        end_epoch,
        3600.0,
    )
    body_settings = environment_setup.get_default_body_settings_time_limited(
        bodies_to_create,
        start_epoch,
        end_epoch,
        global_frame_origin,
        global_frame_orientation,
    )

    body_settings.get("Earth").shape_settings = (
        environment_setup.shape.oblate_spherical_spice()
    )
    body_settings.get("Earth").rotation_model_settings = (
        environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006,
            global_frame_orientation,
            hourly_interpolator,
            hourly_interpolator,
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                start_epoch,
                end_epoch,
                60.0,
            ),
        )
    )
    body_settings.get("Earth").gravity_field_settings.associated_reference_frame = (
        "ITRS"
    )
    body_settings.get("Earth").ground_station_settings = (
        environment_setup.ground_station.dsn_stations()
    )

    body_settings.get("Mars").rotation_model_settings = (
        environment_setup.rotation_model.mars_high_accuracy(
            base_frame=global_frame_orientation
        )
    )
    body_settings.get("Mars").gravity_field_settings = (
        environment_setup.gravity_field.predefined_spherical_harmonic(
            environment_setup.gravity_field.jgmro120d, 120
        )
    )
    body_settings.get("Mars").gravity_field_settings.associated_reference_frame = (
        "Mars_Fixed"
    )
    body_settings.get("Mars").gravity_field_variation_settings = [
        environment_setup.gravity_field_variation.solid_body_tide("Sun", 0.1697, 2),
        environment_setup.gravity_field_variation.solid_body_tide("Phobos", 0.1697, 2),
    ]
    body_settings.get("Mars").climate_model_settings = (
        environment_setup.atmosphere.mars_climate_database_climate_model(
            mcd_data_path=str(HERE.parents[2] / "third_parties" / "mcd" / "data"),
            dust_scenario=1,
            perturbation_key=0,
            high_resolution_mode=0,
        )
    )
    body_settings.get("Mars").atmosphere_settings = (
        environment_setup.atmosphere.mars_climate_database_atmosphere_model()
    )

    # The mean Martian Bond albedo is approximately 0.25 (NASA Mars Fact Sheet).
    mars_surface_radiosity = [
        environment_setup.radiation_pressure.constant_radiosity(250.0),
        environment_setup.radiation_pressure.constant_albedo_surface_radiosity(
            0.25, "Sun"
        ),
    ]
    body_settings.get("Mars").radiation_source_settings = (
        environment_setup.radiation_pressure.panelled_extended_radiation_source(
            mars_surface_radiosity,
            [6, 12],
        )
    )

    spacecraft_name = "MRO"
    spacecraft_central_body = "Mars"
    body_settings.add_empty_settings(spacecraft_name)
    ephemeris_time_step = 10.0
    # Keep the full Lagrange stencil available throughout the environment interval.
    ephemeris_buffer = 6.0 * ephemeris_time_step
    body_settings.get(spacecraft_name).ephemeris_settings = (
        environment_setup.ephemeris.interpolated_spice(
            start_epoch - ephemeris_buffer,
            end_epoch + ephemeris_buffer,
            ephemeris_time_step,
            spacecraft_central_body,
            global_frame_orientation,
        )
    )
    body_settings.get(spacecraft_name).rotation_model_settings = (
        environment_setup.rotation_model.spice(
            global_frame_orientation, spacecraft_name + "_SPACECRAFT", ""
        )
    )
    body_settings.get(spacecraft_name).constant_mass = 1262.39
    body_settings.get(spacecraft_name).vehicle_shape_settings = macromodel_mro(
        reduced_solar_arrays=False
    )
    body_settings.get(spacecraft_name).aerodynamic_coefficient_settings = (
        environment_setup.aerodynamic_coefficients.constant_variable_cross_section(
            [2.0, 0.0, 0.01],
            0,
        )
    )
    body_settings.get(spacecraft_name).radiation_pressure_target_settings = (
        environment_setup.radiation_pressure.panelled_radiation_target(
            {"Sun": ["Mars"]},
            {"Sun": 0},
        )
    )

    bodies = environment_setup.create_system_of_bodies(body_settings)
    spacecraft_systems = bodies.get(spacecraft_name).system_models
    spacecraft_systems.transponder_delay = 1.4149e-6
    spacecraft_systems.set_default_transponder_turnaround_ratio_function()

    # BodySettings currently accepts one target model. Add the cannonball model
    # used for Mars radiation pressure after body creation as a workaround.
    environment_setup.add_radiation_pressure_target_model(
        bodies,
        spacecraft_name,
        environment_setup.radiation_pressure.cannonball_radiation_target(
            5.0,
            1.5,
            {"Mars": []},
        ),
    )
    return bodies


# =============================================================================================
# ================   CREATE AND PREPROCESS OBSERVATIONS   =====================================
# =============================================================================================


def create_observations(
    tracking_data: Sequence[TrackingData],
    supplementary_data: Sequence[TrackingSupplementaryData],
    bodies: environment.SystemOfBodies,
    arc_start: time_representation.Time,
    arc_end: time_representation.Time,
) -> tuple[
    observations.ObservationCollection,
    time_representation.Time,
    time_representation.Time,
]:
    """Apply supplementary data, select the arc, and compress its Doppler data."""
    print("Preparing and compressing observations...")
    observations.set_tracking_supplementary_data_in_bodies(
        bodies, supplementary_data
    )
    original_observations = (
        observations.create_observation_collection_from_tracking_data(
            tracking_data, bodies
        )
    )

    arc_filter = observations.observations_processing.observation_filter(
        observations.observations_processing.ObservationFilterType.time_bounds_filtering,
        arc_start.to_float(),
        arc_end.to_float(),
        use_opposite_condition=True,
    )
    original_observations.filter_observations(arc_filter)
    original_observations.remove_empty_observation_sets()

    observation_time_limits = original_observations.time_bounds_time_object
    obs_start_time = observation_time_limits[0]
    obs_end_time = observation_time_limits[1]
    compressed_observations = observations.create_compressed_doppler_collection(
        original_observations, 60, 10
    )
    return compressed_observations, obs_start_time, obs_end_time


# =============================================================================================
# ================   DEFINE THE ESTIMATED PARAMETERS   ========================================
# =============================================================================================


def empirical_arc_starts_from_reference_state(
    reference_state: np.ndarray,
    gravitational_parameter: float,
    propagation_start: float,
    propagation_end: float,
) -> list[float]:
    """Return one-orbit empirical boundaries from an unperturbed state."""
    keplerian_state = element_conversion.cartesian_to_keplerian(
        reference_state, gravitational_parameter
    )
    semi_major_axis = keplerian_state[0]
    orbital_period = (
        2.0 * np.pi
        * np.sqrt(semi_major_axis**3 / gravitational_parameter)
    )
    arc_start_times = []
    current_arc_start_time = float(propagation_start)
    while current_arc_start_time < float(propagation_end):
        arc_start_times.append(current_arc_start_time)
        current_arc_start_time += orbital_period
    return arc_start_times


def empirical_arc_starts(
    one_orbit_starts: Sequence[float], arc_index: int
) -> list[float]:
    """Create two-orbit empirical arcs and merge boundaries without data support."""
    if len(one_orbit_starts) < 2:
        raise ValueError("At least two one-orbit starts are needed.")
    arc_index = int(arc_index)
    if arc_index not in range(len(ESTIMATION_ARCS)):
        raise ValueError(
            "The unsupported-edge merge is specific to the seven estimation arcs."
        )
    first = float(one_orbit_starts[0])
    period = float(one_orbit_starts[1] - one_orbit_starts[0])
    end = float(one_orbit_starts[-1] + period)
    starts = list(
        np.arange(
            first,
            end - period + 1.0e-6,
            period * 2.0,
        )
    )
    if len(starts) != 20:
        raise ValueError(
            f"Arc {arc_index} has {len(starts)} two-orbit boundaries; expected 20."
        )

    # These boundaries produce zero design-matrix columns because no retained
    # observations are sensitive to their coefficients. Removing a boundary
    # extends the adjacent empirical-acceleration interval across the data gap.
    if arc_index == 0:
        starts = starts[:-2]
    elif arc_index == 5:
        starts.pop(1)
    return starts


def create_parameter_settings(
    propagator_settings: propagation_setup.propagator.TranslationalStatePropagatorSettings,
    bodies: environment.SystemOfBodies,
    empirical_starts: Sequence[float],
) -> list[parameters_setup.EstimatableParameterSettings]:
    """Create the estimated state, Sun scale, and TN empirical parameters."""
    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)
    parameter_settings.append(
        parameters_setup.radiation_pressure_target_direction_scaling("MRO", "Sun")
    )
    shapes = parameters_setup.EmpiricalAccelerationFunctionalShapes
    components = parameters_setup.EmpiricalAccelerationComponents
    selection = {
        components.along_track_empirical_acceleration_component: [
            shapes.constant_empirical,
            shapes.sine_empirical,
            shapes.cosine_empirical,
        ],
        components.across_track_empirical_acceleration_component: [
            shapes.constant_empirical,
            shapes.sine_empirical,
            shapes.cosine_empirical,
        ],
    }
    parameter_settings.append(
        parameters_setup.arcwise_empirical_accelerations(
            "MRO", "Mars", selection, empirical_starts
        )
    )
    return parameter_settings


def create_parameter_metadata(
    parameter_set: parameters.EstimatableParameterSet,
    empirical_starts: Sequence[float],
) -> ParameterMetadata:
    """Validate native parameter blocks and provide readable index labels."""
    parameter_types = parameters_setup.EstimatableParameterTypes
    metadata = []

    def require_block(parameter_type: Any, expected_size: int) -> int:
        identifier = (parameter_type, ("", ""))
        blocks = parameter_set.indices_for_parameter_type(identifier)
        parameters = parameter_set.parameters_for_parameter_type(identifier)
        if len(blocks) != 1 or len(parameters) != 1:
            raise ValueError(f"Expected one native block for {parameter_type}; got {blocks}.")
        start, size = blocks[0]
        if size != expected_size:
            raise ValueError(
                f"Native block {parameter_type} has size {size}; expected {expected_size}."
            )
        return start

    state_names = ("x", "y", "z", "vx", "vy", "vz")
    state_units = ("m", "m", "m", "m/s", "m/s", "m/s")
    state_start = require_block(parameter_types.initial_body_state_type, 6)
    metadata.extend(
        dict(
            index=state_start + index,
            name=name,
            unit=state_units[index],
            group="state",
            subarc_start=None,
        )
        for index, name in enumerate(state_names)
    )

    sun_start = require_block(
        parameter_types.radiation_pressure_target_direction_scaling_factor_type,
        1,
    )
    metadata.append(
        dict(
            index=sun_start,
            name="sun_scale",
            unit="1",
            group="Sun",
            subarc_start=None,
        )
    )

    empirical_size = 6 * len(empirical_starts)
    empirical_start = require_block(
        parameter_types.arc_wise_empirical_acceleration_coefficients_type,
        empirical_size,
    )
    names = [
        f"empirical_{component}_{shape}"
        for epoch in empirical_starts
        for shape in ("constant", "sine", "cosine")
        for component in ("T", "N")
    ]
    for offset, (name, epoch) in enumerate(
        zip(names, np.repeat(empirical_starts, 6))
    ):
        metadata.append(
            dict(
                index=empirical_start + offset,
                name=name,
                unit="m/s^2",
                group=f"empirical {offset // 6:02d}",
                subarc_start=float(epoch),
            )
        )

    metadata.sort(key=lambda row: row["index"])
    if [row["index"] for row in metadata] != list(
        range(parameter_set.parameter_set_size)
    ):
        raise ValueError("Parameter metadata does not cover every estimated index.")
    return metadata


# =============================================================================================
# ================   CREATE DYNAMICS AND PROPAGATION SETTINGS   ===============================
# =============================================================================================


def create_propagator_settings(
    bodies: environment.SystemOfBodies,
    estimation_epoch: time_representation.Time,
    propagation_start: time_representation.Time,
    propagation_end: time_representation.Time,
) -> tuple[
    propagation_setup.propagator.TranslationalStatePropagatorSettings,
    np.ndarray,
]:
    """Propagate forward and backward from the state estimated at the arc midpoint."""
    print("Setting up accelerations and propagation...")
    accelerations_settings_spacecraft = dict(
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.paneled_target
            ),
        ],
        Mars=[
            propagation_setup.acceleration.spherical_harmonic_gravity(
                120, 120
            ),
            propagation_setup.acceleration.aerodynamic(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.cannonball_target
            ),
            propagation_setup.acceleration.empirical(),
        ],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Phobos=[propagation_setup.acceleration.point_mass_gravity()],
        Deimos=[propagation_setup.acceleration.point_mass_gravity()],
    )
    acceleration_settings = {"MRO": accelerations_settings_spacecraft}
    bodies_to_propagate = ["MRO"]
    central_bodies = ["Mars"]
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_representation.Time(0, 30.0),
        propagation_setup.integrator.rkf_56,
    )
    initial_state = propagation.get_state_of_bodies(
        bodies_to_propagate, central_bodies, bodies, estimation_epoch
    )
    termination_settings = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(propagation_end.to_float()),
        propagation_setup.propagator.time_termination(propagation_start.to_float()),
    )
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        estimation_epoch,
        integrator_settings,
        termination_settings,
    )
    # This interval is measured in propagation time and is retained by the
    # variational-equations solver used during estimation.
    propagator_settings.print_settings.results_print_frequency_in_seconds = 7200.0
    propagator_settings.print_settings.results_print_frequency_in_steps = 0
    return propagator_settings, initial_state


# =============================================================================================
# ================   DEFINE THE A PRIORI PARAMETER CONSTRAINTS   ==============================
# =============================================================================================


def create_inverse_apriori_covariance(
    parameter_set: parameters.EstimatableParameterSet,
) -> np.ndarray:
    """Create checked parameter priors from Tudat's parameter block metadata."""
    parameter_types = parameters_setup.EstimatableParameterTypes
    prior_sigmas = np.full(parameter_set.parameter_set_size, np.nan)

    def set_prior(
        parameter_type: Any, sigma_factory: Callable[[int], Sequence[float]]
    ) -> None:
        identifier = (parameter_type, ("", ""))
        index_blocks = parameter_set.indices_for_parameter_type(identifier)
        parameters = parameter_set.parameters_for_parameter_type(identifier)
        if len(index_blocks) != len(parameters):
            raise ValueError(f"Inconsistent parameter metadata for {parameter_type}.")

        for (start_index, block_size), parameter in zip(index_blocks, parameters):
            block_sigmas = np.asarray(sigma_factory(block_size), dtype=float)
            if block_sigmas.size != block_size:
                raise ValueError(
                    f"Prior for {parameter.parameter_description} has size "
                    f"{block_sigmas.size}; expected {block_size}."
                )
            block_slice = slice(start_index, start_index + block_size)
            if np.isfinite(prior_sigmas[block_slice]).any():
                raise ValueError(
                    f"Overlapping a priori parameter block: "
                    f"{parameter.parameter_description}."
                )
            prior_sigmas[block_slice] = block_sigmas
            sigma_summary = ", ".join(
                f"{value:.1e}" for value in np.unique(block_sigmas)
            )
            print(
                f"Prior block [{start_index}:{start_index + block_size}]: "
                f"{parameter.parameter_description}; sigma = {sigma_summary}"
            )

    def initial_state_sigmas(block_size: int) -> np.ndarray:
        if block_size % 6:
            raise ValueError(
                f"Initial-state parameter block has unexpected size {block_size}."
            )
        return np.tile(
            [100.0] * 3 + [0.1] * 3,
            block_size // 6,
        )

    set_prior(
        parameter_types.initial_body_state_type,
        initial_state_sigmas,
    )
    set_prior(
        parameter_types.radiation_pressure_target_direction_scaling_factor_type,
        lambda block_size: np.full(block_size, 0.2),
    )
    set_prior(
        parameter_types.arc_wise_empirical_acceleration_coefficients_type,
        lambda block_size: np.full(block_size, 3.0e-6),
    )

    if not np.isfinite(prior_sigmas).all():
        missing_indices = np.flatnonzero(~np.isfinite(prior_sigmas))
        raise ValueError(f"Missing a priori uncertainties at indices {missing_indices}.")
    return np.diag(np.reciprocal(np.square(prior_sigmas)))


# =============================================================================================
# ================   CONFIGURE THE DOPPLER MODEL AND FILTER THE DATA   ========================
# =============================================================================================


def set_antenna_reference_point(
    observation_collection: observations.ObservationCollection,
    bodies: environment.SystemOfBodies,
) -> None:
    """Set the MRO antenna position relative to its centre of mass."""
    print("Setting the MRO antenna reference point...")
    centre_of_mass = np.array([-0.001235, -1.14978, -0.001288])
    position_history = {}
    for observation_times in observation_collection.get_observation_times_objects():
        epoch = observation_times[0].to_float() - 3600.0
        final_epoch = observation_times[-1].to_float() + 3600.0
        while epoch <= final_epoch:
            state = np.zeros(6)
            state[:3] = spice.get_body_cartesian_position_at_epoch(
                "-74214", "-74000", "MRO_SPACECRAFT", "none", epoch
            ) - centre_of_mass
            position_history[epoch] = state
            epoch += 60.0

    antenna = environment_setup.ephemeris.create_ephemeris(
        environment_setup.ephemeris.tabulated(
            position_history, "-74000", "MRO_SPACECRAFT"
        ),
        "Antenna",
    )
    observation_collection.set_reference_point(
        bodies,
        antenna,
        "Antenna",
        "MRO",
        observable_models_setup.links.LinkEndType.reflector1,
    )


def create_observation_models(
    observation_collection: observations.ObservationCollection,
    bodies: environment.SystemOfBodies,
    tro_files: Sequence[str],
    ion_files: Sequence[str],
) -> tuple[list[observable_models_setup.model_settings.ObservationModelSettings], list[Any]]:
    """Create corrected DSN averaged-Doppler models and simulators."""
    print("Setting up Doppler observation models...")
    corrections = [
        observable_models_setup.light_time_corrections.approximated_second_order_relativistic_light_time_correction(
            ["Sun"]
        ),
        observable_models_setup.light_time_corrections.dsn_tabulated_tropospheric_light_time_correction(
            tro_files
        ),
        observable_models_setup.light_time_corrections.dsn_tabulated_ionospheric_light_time_correction(
            ion_files, {74: "MRO"}
        ),
    ]
    observable_type = observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type
    model_settings = [
        observable_models_setup.model_settings.dsn_n_way_doppler_averaged(
            link_definition, corrections
        )
        for link_definition in observation_collection.link_definitions_per_observable[
            observable_type
        ]
    ]
    simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        model_settings, bodies
    )
    return model_settings, simulators


def filter_prefit_residuals(
    observation_collection: observations.ObservationCollection,
    simulators: Sequence[Any],
    bodies: environment.SystemOfBodies,
) -> pd.DataFrame:
    """Compute SPICE-trajectory residuals and apply the 8 mHz outlier filter."""
    print("Computing and filtering prefit residuals...")
    observations.compute_residuals_and_dependent_variables(
        observation_collection, simulators, bodies
    )
    unfiltered = np.asarray(observation_collection.get_concatenated_residuals()).copy()
    observable_type = observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type
    parser = observations.observations_processing.observation_parser(observable_type)
    observation_collection.filter_observations(
        {
            parser: observations.observations_processing.observation_filter(
                observations.observations_processing.ObservationFilterType.residual_filtering,
                0.008,
            )
        }
    )

    residuals = observation_collection.get_concatenated_residuals(parser)
    link_ids = observation_collection.get_concatenated_link_definition_ids(parser)
    link_definitions = observation_collection.link_definition_ids
    link_ends = [
        " - ".join(
            (
                link_definitions[link_id][
                    observable_models_setup.links.LinkEndType.transmitter
                ].reference_point,
                link_definitions[link_id][
                    observable_models_setup.links.LinkEndType.receiver
                ].reference_point,
            )
        )
        for link_id in link_ids
    ]
    table = pd.DataFrame(
        {
            "spice": residuals,
            "time": observation_collection.get_concatenated_observation_times(parser),
            "link_id": [int(link_id) for link_id in link_ids],
            "link_ends": link_ends,
            "msrType": "doppler",
        }
    )
    table.attrs.update(
        unfiltered_count=len(unfiltered),
        unfiltered_rms_hz=float(np.sqrt(np.mean(unfiltered**2))),
        unfiltered_max_hz=float(np.max(np.abs(unfiltered))),
    )
    return table


# =============================================================================================
# ================   DEFINE ESTIMATION SETTINGS AND PERFORM THE FIT   =========================
# =============================================================================================


def estimate_parameters(
    bodies: environment.SystemOfBodies,
    observation_collection: observations.ObservationCollection,
    observation_model_settings: Sequence[
        observable_models_setup.model_settings.ObservationModelSettings
    ],
    propagator_settings: propagation_setup.propagator.TranslationalStatePropagatorSettings,
    parameters_to_estimate: parameters.EstimatableParameterSet,
    inverse_apriori_covariance: np.ndarray,
    arc_index: int,
) -> estimation_analysis.EstimationOutput:
    """Run the constrained estimation and return its native output."""
    print("Running estimation...")
    observation_collection.set_constant_weight(1.0)
    estimation_input = estimation_analysis.EstimationInput(
        observation_collection,
        inverse_apriori_covariance=inverse_apriori_covariance,
        convergence_checker=estimation_analysis.estimation_convergence_checker(5),
        apply_apriori_parameter_deviation=True,
    )
    estimation_input.define_estimation_settings(
        reintegrate_equations_on_first_iteration=False,
        reintegrate_variational_equations=False,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
        limit_condition_number_for_warning=1.0,
        condition_number_warning_each_iteration=True,
    )
    output = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_model_settings,
        propagator_settings,
        integrate_on_creation=True,
    ).perform_estimation(estimation_input)
    failed = (
        output.exception_during_propagation
        or output.exception_during_inversion
        or any(
            not result.dynamics_results.integration_completed_successfully
            for result in output.simulation_results_per_iteration
        )
    )
    if failed:
        raise RuntimeError(f"MRO TNF estimation failed for arc {arc_index}")
    return output


# =============================================================================================
# ================   PROCESS ONE ESTIMATION ARC   =============================================
# =============================================================================================


def run_arc(arc_index: int) -> ArcResult:
    """Fit one MRO arc and return only picklable selected products."""
    plt.switch_backend("Agg")
    start_text, end_text = ESTIMATION_ARCS[arc_index]
    start_datetime = datetime.fromisoformat(start_text)
    end_datetime = datetime.fromisoformat(end_text)
    (
        tnf_files,
        clock_files,
        orientation_files,
        tro_files,
        ion_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    ) = prepare_arc_inputs(
        arc_index,
        start_datetime,
        end_datetime,
    )
    print(f"Processing arc {arc_index}: {start_datetime} to {end_datetime}")

    load_spice_kernels(
        clock_files,
        orientation_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    )
    tracking_data, supplementary_data = load_tracking_data(tnf_files)

    # Define arc time interval
    arc_start = time_representation.DateTime.from_python_datetime(
        start_datetime
    ).to_epoch()
    arc_end = time_representation.DateTime.from_python_datetime(end_datetime).to_epoch()

    time_scale_converter = time_representation.default_time_scale_converter()
    arc_start = time_scale_converter.convert_time_object(
        input_scale=time_representation.utc_scale,
        output_scale=time_representation.tdb_scale,
        input_value=time_representation.Time(arc_start),
    )
    arc_end = time_scale_converter.convert_time_object(
        input_scale=time_representation.utc_scale,
        output_scale=time_representation.tdb_scale,
        input_value=time_representation.Time(arc_end),
    )
    estimation_epoch = arc_start + (arc_end - arc_start) / 2.0
    print(
        "Estimating the state at "
        f"{time_representation.DateTime.from_epoch(estimation_epoch).to_python_datetime()} TDB"
    )

    # Keep one extra hour of environment coverage beyond the padded propagation
    # interval, including for the initial interpolated-SPICE state.
    environment_start_time = arc_start - 7200.0
    environment_end_time = arc_end + 7200.0

    bodies = create_environment(environment_start_time, environment_end_time)
    compressed_observations, _, _ = create_observations(
        tracking_data,
        supplementary_data,
        bodies,
        arc_start,
        arc_end,
    )
    prop_start_time = arc_start - 3600.0
    prop_end_time = arc_end + 3600.0

    set_antenna_reference_point(compressed_observations, bodies)
    observation_model_settings, observation_simulators = create_observation_models(
        compressed_observations, bodies, tro_files, ion_files
    )
    residDf = filter_prefit_residuals(
        compressed_observations, observation_simulators, bodies
    )
    propagator_settings, unperturbed_initial_state = create_propagator_settings(
        bodies,
        estimation_epoch,
        prop_start_time,
        prop_end_time,
    )

    # =========================================================================================
    # ================   DEFINE THE PARAMETERS TO ESTIMATE   ================================
    # =========================================================================================

    print("Setting up estimated parameters...")

    # Define arc start times for arc-wise empirical accelerations
    mars_gravitational_parameter = bodies.get("Mars").gravitational_parameter
    one_orbit_arc_start_times = empirical_arc_starts_from_reference_state(
        unperturbed_initial_state,
        mars_gravitational_parameter,
        prop_start_time.to_float(),
        prop_end_time.to_float(),
    )
    empirical_arc_start_times = empirical_arc_starts(
        one_orbit_arc_start_times, arc_index
    )
    parameter_settings = create_parameter_settings(
        propagator_settings, bodies, empirical_arc_start_times
    )

    # Create set of parameters to estimate
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )
    parameter_metadata = create_parameter_metadata(
        parameters_to_estimate, empirical_arc_start_times
    )
    inverse_apriori_covariance = create_inverse_apriori_covariance(
        parameters_to_estimate
    )
    initial_parameters = parameters_to_estimate.parameter_vector.copy()

    estimation_output = estimate_parameters(
        bodies,
        compressed_observations,
        observation_model_settings,
        propagator_settings,
        parameters_to_estimate,
        inverse_apriori_covariance,
        arc_index,
    )

    print("Processing estimation results...")
    best_iteration = estimation_output.best_iteration
    residDf["prefit"] = estimation_output.residual_history[:, 0]
    residDf["postfit"] = estimation_output.residual_history[:, best_iteration]

    prefit_state_history = estimation_output.simulation_results_per_iteration[
        0
    ].dynamics_results.state_history_float

    estimated_state_history = estimation_output.simulation_results_per_iteration[
        best_iteration
    ].dynamics_results.state_history_float
    print(f"Finished processing arc {arc_index}")
    return _plain_arc_result({
        "arc_index": arc_index,
        "residuals": residDf,
        "estimation_output": estimation_output,
        "prefit_state_history": prefit_state_history,
        "postfit_state_history": estimated_state_history,
        "initial_parameters": initial_parameters,
        "estimation_epoch": estimation_epoch.to_float(),
        "arc_bounds": (arc_start.to_float(), arc_end.to_float()),
        "empirical_arc_start_times": empirical_arc_start_times,
        "parameter_metadata": parameter_metadata,
        "inverse_apriori_covariance": inverse_apriori_covariance,
    })


# =============================================================================================
# ================   COMPARE THE FITTED ORBITS TO THE SPICE TRAJECTORY   ======================
# =============================================================================================


def compare_history_to_spice(
    history: StateHistory, start: float, end: float, step_seconds: float
) -> pd.DataFrame:
    """Return estimated-minus-SPICE position in the SPICE-reference RTN frame."""
    import spiceypy

    epochs = np.arange(float(start), float(end) + 1.0e-7, step_seconds)
    if start - min(history) < 8 * step_seconds or max(history) - end < 8 * step_seconds:
        raise ValueError("The score grid is too close to a propagated-history edge.")
    interpolation = interpolators.create_one_dimensional_vector_interpolator(
        history, interpolators.lagrange_interpolation(8)
    )
    states = np.array(
        [interpolation.interpolate(float(epoch)).reshape(6) for epoch in epochs]
    )
    reference = np.array(
        [
            spiceypy.spkezr("-74", float(epoch), "J2000", "NONE", "499")[0]
            for epoch in epochs
        ]
    ) * 1000.0
    radial = reference[:, :3] / np.linalg.norm(reference[:, :3], axis=1)[:, None]
    normal = np.cross(reference[:, :3], reference[:, 3:])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    transverse = np.cross(normal, radial)
    delta = states[:, :3] - reference[:, :3]
    rtn = np.column_stack(
        [
            np.einsum("ij,ij->i", delta, direction)
            for direction in (radial, transverse, normal)
        ]
    )
    np.testing.assert_allclose(
        np.linalg.norm(rtn, axis=1), np.linalg.norm(delta, axis=1), atol=1.0e-9
    )
    return pd.DataFrame(
        {"t": epochs, "R": rtn[:, 0], "T": rtn[:, 1], "N": rtn[:, 2]}
    )


def retained_observation_tag_orbit(
    orbit: pd.DataFrame, residuals: pd.DataFrame
) -> pd.DataFrame:
    """Apply the primary inclusive first/last retained-observation-tag mask."""
    first = float(residuals["time"].min())
    last = float(residuals["time"].max())
    selected = orbit[(orbit["t"] >= first) & (orbit["t"] <= last)].copy()
    if selected.empty:
        raise ValueError("The retained-observation-tag orbit mask is empty.")
    return selected


def _plain_arc_result(arc_result: ArcResult) -> ArcResult:
    """Extract only picklable selected-iteration products from one native result."""
    output = arc_result["estimation_output"]
    best_iteration = int(output.best_iteration)
    residuals = arc_result["residuals"].copy()
    start, end = arc_result["arc_bounds"]
    prefit_orbit = retained_observation_tag_orbit(
        compare_history_to_spice(
            arc_result["prefit_state_history"],
            start,
            end,
            60.0,
        ),
        residuals,
    )
    postfit_orbit = retained_observation_tag_orbit(
        compare_history_to_spice(
            arc_result["postfit_state_history"],
            start,
            end,
            60.0,
        ),
        residuals,
    )
    arc_index = int(arc_result["arc_index"])
    residuals["arc_index"] = arc_index
    prefit_orbit["arc_index"] = arc_index
    postfit_orbit["arc_index"] = arc_index

    metadata = [dict(row) for row in arc_result["parameter_metadata"]]
    parameter_history = np.asarray(output.parameter_history, dtype=float)
    selected_parameters = parameter_history[:, best_iteration]
    if len(metadata) != selected_parameters.size:
        raise ValueError("Parameter labels do not match the selected native vector.")
    initial_parameters = np.asarray(arc_result["initial_parameters"], dtype=float)
    estimation_epoch = float(arc_result["estimation_epoch"])
    empirical_starts = np.asarray(
        arc_result["empirical_arc_start_times"], dtype=float
    )
    empirical_ends = np.r_[empirical_starts[1:], float(end)]
    empirical_intervals = {
        float(interval_start): (
            max(float(interval_start), float(start)),
            min(float(interval_end), float(end)),
        )
        for interval_start, interval_end in zip(empirical_starts, empirical_ends)
    }
    for row, initial, value in zip(metadata, initial_parameters, selected_parameters):
        row.update(
            arc_index=arc_index,
            initial=float(initial),
            value=float(value),
            delta=float(value - initial),
        )
        if row["group"] == "state":
            # These are corrections to the single midpoint state, not a
            # continuous Cartesian-state history.
            row.update(
                validity_start=estimation_epoch,
                validity_end=estimation_epoch,
            )
        elif row["subarc_start"] is None:
            row.update(validity_start=float(start), validity_end=float(end))
        else:
            validity_start, validity_end = empirical_intervals[
                float(row["subarc_start"])
            ]
            if validity_end <= validity_start:
                raise ValueError(
                    f"Parameter {row['name']} has an empty validity interval."
                )
            row.update(
                validity_start=validity_start,
                validity_end=validity_end,
            )

    correlations = np.asarray(output.correlations, dtype=float).copy()
    if correlations.shape != (len(metadata), len(metadata)):
        raise ValueError("Native correlation dimensions do not match parameters.")
    return {
        "arc_index": arc_index,
        "arc_bounds": (float(start), float(end)),
        "best_iteration": best_iteration,
        "residuals": residuals,
        "prefit_orbit": prefit_orbit,
        "postfit_orbit": postfit_orbit,
        "parameters": metadata,
        "correlations": correlations,
        "empirical_arc_start_times": empirical_starts,
        "inverse_apriori_diagonal": np.diag(
            arc_result["inverse_apriori_covariance"]
        ).copy(),
    }


def _pooled_primary_metrics(results: Sequence[ArcResult]) -> dict[str, float]:
    """Compute pooled primary-mask orbit and residual statistics."""
    residuals = pd.concat([result["residuals"] for result in results])
    orbit = pd.concat([result["postfit_orbit"] for result in results])
    position = orbit[["R", "T", "N"]].to_numpy()
    return {
        "residual_rms_mhz": float(
            np.sqrt(np.mean(np.square(residuals["postfit"]))) * 1.0e3
        ),
        "residual_max_mhz": float(np.max(np.abs(residuals["postfit"])) * 1.0e3),
        **{
            f"{component}_rms_m": float(
                np.sqrt(np.mean(np.square(orbit[component])))
            )
            for component in "RTN"
        },
        "position_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(position), axis=1)))
        ),
        "position_max_m": float(np.max(np.linalg.norm(position, axis=1))),
    }


# =============================================================================================
# ================   PLOT THE ESTIMATION RESULTS   ============================================
# =============================================================================================


def plot_results(results: Sequence[ArcResult]) -> list[Figure]:
    """Create interactive multi-arc residual, orbit, parameter, and correlation plots."""
    results = sorted(results, key=lambda result: result["arc_index"])
    results_by_arc = {int(result["arc_index"]): result for result in results}
    if len(results_by_arc) != len(results):
        raise ValueError("Each plotted result must have a unique arc_index.")
    residuals = pd.concat([result["residuals"] for result in results])
    prefit_orbit = pd.concat([result["prefit_orbit"] for result in results])
    postfit_orbit = pd.concat([result["postfit_orbit"] for result in results])
    origin = min(result["arc_bounds"][0] for result in results)
    figures = []

    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, column, title in zip(
        axes,
        ("spice", "prefit", "postfit"),
        (
            "Reconstructed-SPICE residual",
            "Initial propagated residual",
            "Selected postfit residual",
        ),
    ):
        for arc_index, frame in residuals.groupby("arc_index", sort=True):
            axis.plot(
                (frame["time"] - origin) / 86400.0,
                frame[column] * 1.0e3,
                ".",
                markersize=2,
                label=f"arc {arc_index}",
            )
        axis.set_ylabel("Doppler [mHz]")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=7, fontsize=8)
    axes[-1].set_xlabel("TDB days since first estimation arc start")
    figure.suptitle("MRO DSN Doppler residuals")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figures.append(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, (stage, orbit) in enumerate(
        (("Initial propagated", prefit_orbit), ("Selected postfit", postfit_orbit))
    ):
        for column, component in enumerate("RTN"):
            axis = axes[row, column]
            for arc_index, frame in orbit.groupby("arc_index", sort=True):
                axis.plot(
                    (frame["t"] - origin) / 86400.0,
                    frame[component],
                    ".",
                    markersize=2,
                    label=f"arc {arc_index}",
                )
            rms = np.sqrt(np.mean(np.square(orbit[component])))
            axis.set_title(f"{stage} {component}; RMS={rms:.3f} m")
            axis.set_ylabel("estimated - SPICE [m]")
            axis.grid(alpha=0.3)
            if row == 1:
                axis.set_xlabel("TDB days since first estimation arc start")
    axes[0, 0].legend(ncol=1, fontsize=7)
    figure.suptitle(
        "MRO orbit difference on retained-tag grids (SPICE-reference RTN basis)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figures.append(figure)

    parameter_rows = pd.DataFrame(
        [row for result in results for row in result["parameters"]]
    )
    plotted_parameters = [
        "sun_scale",
        "empirical_T_constant",
        "empirical_T_sine",
        "empirical_T_cosine",
        "empirical_N_constant",
        "empirical_N_sine",
        "empirical_N_cosine",
    ]
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False)
    for axis, name in zip(axes.flat, plotted_parameters):
        selected = parameter_rows[parameter_rows["name"] == name]
        for arc_index, frame in selected.groupby("arc_index", sort=True):
            arc_bounds = results_by_arc[int(arc_index)]["arc_bounds"]
            frame = frame.sort_values("validity_start")
            interval_starts = frame["validity_start"].fillna(arc_bounds[0]).to_numpy(
                dtype=float
            )
            interval_end = float(frame["validity_end"].fillna(arc_bounds[1]).iloc[-1])
            values = frame["value"].to_numpy(dtype=float)
            axis.step(
                (np.r_[interval_starts, interval_end] - origin) / 86400.0,
                np.r_[values, values[-1]],
                where="post",
                linewidth=1.25,
                label=f"arc {arc_index}",
            )
        axis.set_title(name)
        axis.set_ylabel("scale [1]" if name == "sun_scale" else "coefficient [m/s²]")
        axis.grid(alpha=0.3)
        axis.set_xlabel("TDB days since first estimation arc start")
    for axis in list(axes.flat)[len(plotted_parameters):]:
        axis.set_visible(False)
    axes.flat[0].legend(ncol=2, fontsize=7)
    figure.suptitle("Selected fitted Sun scale and two-orbit TN coefficients")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figures.append(figure)

    state_names = ("x", "y", "z", "vx", "vy", "vz")
    state_units = ("m", "m", "m", "m/s", "m/s", "m/s")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, squeeze=False)
    for axis, name, unit in zip(axes.flat, state_names, state_units):
        selected = parameter_rows[parameter_rows["name"] == name].sort_values(
            "arc_index"
        )
        axis.plot(
            (selected["validity_start"].astype(float) - origin) / 86400.0,
            selected["delta"],
            "o",
            markersize=6,
        )
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        axis.set_title(name)
        axis.set_ylabel(f"midpoint correction [{unit}]")
        axis.set_xlabel("TDB days since first estimation arc start")
        axis.grid(alpha=0.3)
    figure.suptitle(
        "Estimated Cartesian initial-state corrections at each arc midpoint "
        "(not continuous states)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figures.append(figure)

    for result in results:
        correlations = result["correlations"]
        metadata = result["parameters"]
        groups = []
        for index, row in enumerate(metadata):
            if not groups or groups[-1][0] != row["group"]:
                groups.append([row["group"], index, index])
            else:
                groups[-1][2] = index
        ticks = [(first + last) / 2 for _, first, last in groups]
        labels = [name.replace("empirical ", "TN ") for name, _, _ in groups]
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(
            correlations,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_xticks(ticks, labels, rotation=90, fontsize=7)
        axis.set_yticks(ticks, labels, fontsize=7)
        for _, first, _ in groups[1:]:
            axis.axhline(first - 0.5, color="black", linewidth=0.25, alpha=0.4)
            axis.axvline(first - 0.5, color="black", linewidth=0.25, alpha=0.4)
        axis.set_title(
            f"Arc {result['arc_index']} selected posterior correlations including "
            f"priors (native iteration {result['best_iteration']})"
        )
        axis.set_xlabel(
            "State order: x, y, z, vx, vy, vz. TN block order: T constant, "
            "N constant, T sine, N sine, T cosine, N cosine. Includes prior "
            "information."
        )
        figure.colorbar(image, ax=axis, label="signed correlation")
        figure.tight_layout()
        figures.append(figure)
    return figures


# =============================================================================================
# ================   RUN THE SEVEN ESTIMATION ARCS IN PARALLEL   ==============================
# =============================================================================================


def run_estimation() -> None:
    """Estimate seven independent arcs in parallel and plot the combined results."""
    started = t.time()
    print(
        "Running seven MRO arcs in isolated processes: RKF56/30 s, "
        "anchored priors, two-orbit edge-merged TN empiricals, fixed drag/lift scales."
    )
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(ESTIMATION_ARCS), mp_context=context
    ) as executor:
        futures = {
            executor.submit(run_arc, index): index
            for index in range(len(ESTIMATION_ARCS))
        }
        for future in as_completed(futures):
            arc_index = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"Arc {arc_index} complete; selected native iteration "
                f"{result['best_iteration']}."
            )

    results.sort(key=lambda result: result["arc_index"])
    metrics = _pooled_primary_metrics(results)
    print(
        "Primary retained-tag result: Doppler RMS/max "
        f"{metrics['residual_rms_mhz']:.6f}/{metrics['residual_max_mhz']:.6f} mHz; "
        "R/T/N/3D RMS "
        f"{metrics['R_rms_m']:.6f}/{metrics['T_rms_m']:.6f}/"
        f"{metrics['N_rms_m']:.6f}/{metrics['position_rms_m']:.6f} m."
    )
    plot_results(results)
    print(f"Total runtime: {t.time() - started:.2f} s")
    plt.show()


if __name__ == "__main__":
    run_estimation()
