# %%
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


from mro_utils import macromodel_mro, get_rsw_state_difference

from tudatpy.data_input.environment_data import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.data_input.tracking_data.tnf import (
    OpenRampHandling,
    read_tnf_data,
)

from tudatpy.dynamics import (
    environment_setup,
    propagation_setup,
    parameters_setup,
    propagation,
)
from tudatpy import estimation
from tudatpy.estimation import (
    estimation_analysis,
    observable_models_setup,
    observations,
    observations_setup,
)

from tudatpy.math import interpolators

import time as t


def load_spice_kernels(
    clock_files,
    orientation_files,
    trajectory_files,
    frames_def_file,
    structure_file,
):
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


def load_tracking_data(tnf_files):
    """Load MRO Doppler observations and their supplementary data."""
    print("Loading TNF tracking data...")
    return read_tnf_data(
        tnf_files,
        ["doppler"],
        spacecraft_name="MRO",
        open_ramp_handling=OpenRampHandling.close_silently,
    )


def create_environment(environment_start_time, environment_end_time):
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
    body_settings = environment_setup.get_default_body_settings_time_limited(
        bodies_to_create,
        environment_start_time.to_float(),
        environment_end_time.to_float(),
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
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                environment_start_time.to_float(),
                environment_end_time.to_float(),
                3600.0,
            ),
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                environment_start_time.to_float(),
                environment_end_time.to_float(),
                3600.0,
            ),
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                environment_start_time.to_float(),
                environment_end_time.to_float(),
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
    atmosphere_model = os.environ.get("MRO_ATMOSPHERE_MODEL", "mcd").lower()
    if atmosphere_model == "mcd":
        body_settings.get("Mars").climate_model_settings = (
            environment_setup.atmosphere.mars_climate_database_climate_model(
                mcd_data_path=os.environ.get("MRO_MCD_DATA_PATH", ""),
                dust_scenario=int(os.environ.get("MRO_MCD_DUST_SCENARIO", "1")),
                perturbation_key=0,
                high_resolution_mode=int(os.environ.get("MRO_MCD_HIGH_RESOLUTION", "0")),
            )
        )
        body_settings.get("Mars").atmosphere_settings = (
            environment_setup.atmosphere.mars_climate_database_atmosphere_model()
        )
    elif atmosphere_model == "dtm":
        body_settings.get("Mars").atmosphere_settings = (
            environment_setup.atmosphere.mars_dtm()
        )
    else:
        raise ValueError(f"Unsupported Mars atmosphere model: {atmosphere_model}")

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
            environment_start_time.to_float() - ephemeris_buffer,
            environment_end_time.to_float() + ephemeris_buffer,
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
    use_reduced_macromodel = os.environ.get(
        "MRO_REDUCED_SOLAR_ARRAY_MACROMODEL", "0"
    ).lower() in {"1", "true", "yes"}
    body_settings.get(spacecraft_name).vehicle_shape_settings = macromodel_mro(
        reduced_solar_arrays=use_reduced_macromodel
    )

    drag_coefficient = 2.0
    lift_coefficient = float(os.environ.get("MRO_LIFT_COEFFICIENT", "0.01"))
    aerodynamic_model = os.environ.get(
        "MRO_AERODYNAMIC_COEFFICIENT_MODEL", "variable_cross_section"
    ).lower()
    self_shadowing_pixels = os.environ.get("MRO_SELF_SHADOWING_PIXELS", "0")
    aerodynamic_self_shadowing_pixels = int(
        os.environ.get(
            "MRO_AERODYNAMIC_SELF_SHADOWING_PIXELS",
            os.environ.get("MRO_SELF_SHADOWING_PIXELS", "0"),
        )
    )
    radiation_self_shadowing_pixels = int(
        os.environ.get("MRO_RADIATION_SELF_SHADOWING_PIXELS", self_shadowing_pixels)
    )
    sun_radiation_self_shadowing_pixels = int(
        os.environ.get(
            "MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS",
            radiation_self_shadowing_pixels,
        )
    )
    mars_radiation_self_shadowing_pixels = int(
        os.environ.get(
            "MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS",
            radiation_self_shadowing_pixels,
        )
    )
    if aerodynamic_model == "variable_cross_section":
        body_settings.get(spacecraft_name).aerodynamic_coefficient_settings = (
            environment_setup.aerodynamic_coefficients.constant_variable_cross_section(
                [drag_coefficient, 0, lift_coefficient],
                aerodynamic_self_shadowing_pixels,
            )
        )
    elif aerodynamic_model == "storch":
        # MRO DSMC analyses used fully diffuse reflection with full accommodation.
        # Storch avoids the terrestrial-air gas constant fixed in the Sentman model.
        storch_model = (
            environment_setup.aerodynamic_coefficients.GasSurfaceInteractionModelType.storch
        )
        body_settings.get(spacecraft_name).aerodynamic_coefficient_settings = (
            environment_setup.aerodynamic_coefficients.panelled(
                storch_model,
                reference_area=5.0,
                maximum_number_of_pixels=aerodynamic_self_shadowing_pixels,
            )
        )
    else:
        raise ValueError(
            f"Unsupported aerodynamic coefficient model: {aerodynamic_model}"
        )
    mars_panelled_target = os.environ.get("MRO_MARS_RADIATION_TARGET", "cannonball") == "panelled"
    radiation_occultations = {"Sun": ["Mars"]}
    radiation_pixels = {"Sun": sun_radiation_self_shadowing_pixels}
    if mars_panelled_target:
        radiation_occultations["Mars"] = []
        radiation_pixels["Mars"] = mars_radiation_self_shadowing_pixels
    body_settings.get(spacecraft_name).radiation_pressure_target_settings = (
        environment_setup.radiation_pressure.panelled_radiation_target(
            radiation_occultations,
            radiation_pixels,
        )
    )

    bodies = environment_setup.create_system_of_bodies(body_settings)
    spacecraft_systems = bodies.get(spacecraft_name).system_models
    spacecraft_systems.transponder_delay = 1.4149e-6
    spacecraft_systems.set_default_transponder_turnaround_ratio_function()

    # BodySettings currently accepts one target model. Add the cannonball model
    # used for Mars radiation pressure after body creation as a workaround.
    if not mars_panelled_target:
        environment_setup.add_radiation_pressure_target_model(
            bodies,
            spacecraft_name,
            environment_setup.radiation_pressure.cannonball_radiation_target(
                5.0,
                1.5,
                {"Mars": []},
            ),
        )
    return (
        bodies,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    )


def create_observations(tracking_data, supplementary_data, bodies, arc_start, arc_end):
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


def apply_initial_position_offset(initial_state, offset_text=None):
    """Return a copied state with only the configured Cartesian position shifted."""
    if offset_text is None:
        offset_text = os.environ.get("MRO_INITIAL_POSITION_OFFSET_M", "0,0,0")
    try:
        position_offset = np.asarray(
            [float(value.strip()) for value in offset_text.split(",")], dtype=float
        )
    except ValueError as error:
        raise ValueError(
            "MRO_INITIAL_POSITION_OFFSET_M must contain three comma-separated metres"
        ) from error
    if position_offset.shape != (3,) or not np.isfinite(position_offset).all():
        raise ValueError(
            "MRO_INITIAL_POSITION_OFFSET_M must contain three finite Cartesian metres"
        )
    shifted = np.asarray(initial_state, dtype=float).copy()
    shifted[:3] += position_offset
    return shifted


def empirical_arc_starts_from_reference_state(
    reference_state, gravitational_parameter, propagation_start, propagation_end
):
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


def create_propagator_settings(
    bodies,
    spacecraft_name,
    spacecraft_central_body,
    estimation_epoch,
    prop_start_time,
    obs_end_time,
):
    """Propagate forward and backward from the state estimated at the arc midpoint."""
    print("Setting up accelerations and propagation...")
    mars_gravity_degree = int(os.environ.get("MRO_MARS_GRAVITY_DEGREE", "120"))
    integration_step_size = float(
        os.environ.get("MRO_INTEGRATION_STEP_SIZE", "30.0")
    )
    integrator_name = os.environ.get("MRO_INTEGRATOR_COEFFICIENT_SET", "rkf78")
    integrator_coefficients = {
        "rkf56": propagation_setup.integrator.rkf_56,
        "rkf78": propagation_setup.integrator.rkf_78,
    }.get(integrator_name)
    if integrator_coefficients is None:
        raise ValueError(
            "MRO_INTEGRATOR_COEFFICIENT_SET must be 'rkf56' or 'rkf78', "
            f"not {integrator_name!r}."
        )
    accelerations_settings_spacecraft = dict(
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.paneled_target
            ),
        ],
        Mars=[
            propagation_setup.acceleration.spherical_harmonic_gravity(
                mars_gravity_degree, mars_gravity_degree
            ),
            propagation_setup.acceleration.aerodynamic(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.paneled_target
                if os.environ.get("MRO_MARS_RADIATION_TARGET", "cannonball") == "panelled"
                else environment_setup.radiation_pressure.cannonball_target
            ),
            propagation_setup.acceleration.empirical(),
        ],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Phobos=[propagation_setup.acceleration.point_mass_gravity()],
        Deimos=[propagation_setup.acceleration.point_mass_gravity()],
    )
    acceleration_settings = {spacecraft_name: accelerations_settings_spacecraft}
    bodies_to_propagate = [spacecraft_name]
    central_bodies = [spacecraft_central_body]
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_representation.Time(0, integration_step_size),
        integrator_coefficients,
    )
    initial_state = apply_initial_position_offset(propagation.get_state_of_bodies(
        bodies_to_propagate, central_bodies, bodies, estimation_epoch
    ))
    termination_settings = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(obs_end_time.to_float()),
        propagation_setup.propagator.time_termination(prop_start_time.to_float()),
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
    propagator_settings.print_settings.results_print_frequency_in_seconds = float(
        os.environ.get("MRO_PROPAGATION_PRINT_INTERVAL", "7200.0")
    )
    propagator_settings.print_settings.results_print_frequency_in_steps = 0
    return propagator_settings, initial_state


def prepare_arc_inputs(arc_index, start_epoch, end_epoch):
    """Select the local kernels and tracking files needed by the first arc."""
    print("Selecting input files for the arc...")
    kernel_directory = os.path.join(os.path.dirname(__file__), "mro_kernels")
    clock_files = [os.path.join(kernel_directory, "mro_sclkscet_00112_65536.tsc")]
    orientation_files = [
        os.path.join(kernel_directory, file_name)
        for file_name in [
            "mro_sc_psp_111227_120102.bc",
            "mro_hga_psp_111227_120102.bc",
            "mro_sa_psp_111227_120102.bc",
        ]
    ]
    tro_files = [
        os.path.join(kernel_directory, "mromagr2011_335_2012_001.tro"),
        os.path.join(kernel_directory, "mromagr2012_001_2012_032.tro"),
    ]
    ion_files = [
        os.path.join(kernel_directory, "mromagr2011_335_2012_001.ion"),
        os.path.join(kernel_directory, "mromagr2012_001_2012_032.ion"),
    ]
    # This file starts on the preceding UTC day and contains the observations
    # in the requested 12-hour interval.
    tnf_files = [os.path.join(kernel_directory, "mromagr2011_365_1411xmmmv1.tnf")]
    trajectory_files = [os.path.join(kernel_directory, "mro_psp22.bsp")]
    frames_def_file = os.path.join(kernel_directory, "mro_v16.tf")
    structure_file = os.path.join(kernel_directory, "mro_struct_v10.bsp")

    required_files = (
        clock_files
        + orientation_files
        + tro_files
        + ion_files
        + tnf_files
        + trajectory_files
        + [frames_def_file, structure_file]
    )
    missing_files = [file_name for file_name in required_files if not os.path.isfile(file_name)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required MRO input files: " + ", ".join(missing_files)
        )

    return [
        arc_index,
        start_epoch,
        end_epoch,
        tnf_files,
        clock_files,
        orientation_files,
        tro_files,
        ion_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    ]


def plot_spice_residual_diagnostics(residuals):
    """Plot Doppler residuals computed directly from the reconstructed trajectory."""
    residual_rms = np.sqrt(np.mean(np.square(residuals["spice"])))
    residual_maximum = np.max(np.abs(residuals["spice"]))
    reference_epoch = residuals["time"].min()
    relative_time = (residuals["time"] - reference_epoch) / 86400.0
    reference_date = time_representation.DateTime.from_epoch(
        time_representation.Time(reference_epoch)
    ).to_python_datetime()

    figure, axis = plt.subplots(figsize=(14, 6))
    for link_ends, link_residuals in residuals.groupby("link_ends"):
        link_time = (link_residuals["time"] - reference_epoch) / 86400.0
        axis.scatter(
            link_time,
            link_residuals["spice"],
            s=20,
            alpha=0.7,
            label=link_ends,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(
        "Residuals computed with the reconstructed SPICE trajectory\n"
        f"RMS = {residual_rms * 1.0e3:.3f} mHz, "
        f"maximum absolute residual = {residual_maximum * 1.0e3:.3f} mHz"
    )
    axis.set_xlabel(
        f"Time [days since {reference_date.strftime('%Y-%m-%d %H:%M:%S')}]"
    )
    axis.set_ylabel("DSN averaged Doppler residual [Hz]")
    axis.grid(which="both", linestyle="--", linewidth=1.0)
    if residuals["link_ends"].nunique() > 1:
        axis.legend(title="Tracking link")
    figure.suptitle("MRO pre-propagation observation-model diagnostic")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    print(
        f"SPICE residual RMS: {residual_rms * 1.0e3:.3f} mHz; "
        f"maximum: {residual_maximum * 1.0e3:.3f} mHz; "
        f"observations: {len(relative_time)}"
    )
    return figure


def create_inverse_apriori_covariance(parameter_set):
    """Create checked parameter priors from Tudat's parameter block metadata."""
    parameter_types = parameters_setup.EstimatableParameterTypes
    prior_sigmas = np.full(parameter_set.parameter_set_size, np.nan)

    def set_prior(parameter_type, sigma_factory):
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

    def initial_state_sigmas(block_size):
        if block_size % 6:
            raise ValueError(
                f"Initial-state parameter block has unexpected size {block_size}."
            )
        return np.tile([1.0e3] * 3 + [1.0e-1] * 3, block_size // 6)

    set_prior(
        parameter_types.initial_body_state_type,
        initial_state_sigmas,
    )
    for parameter_type in [
        parameter_types.radiation_pressure_target_direction_scaling_factor_type,
        parameter_types.drag_component_scaling_factor_type,
        parameter_types.lift_component_scaling_factor_type,
    ]:
        set_prior(parameter_type, lambda block_size: np.full(block_size, 2.0))
    set_prior(
        parameter_types.arc_wise_empirical_acceleration_coefficients_type,
        lambda block_size: np.full(block_size, 1.0e-6),
    )

    if not np.isfinite(prior_sigmas).all():
        missing_indices = np.flatnonzero(~np.isfinite(prior_sigmas))
        raise ValueError(f"Missing a priori uncertainties at indices {missing_indices}.")
    return np.diag(np.reciprocal(np.square(prior_sigmas)))


def process_arc(
    inputs, *, parameter_builder=None, prior_builder=None, prefit_callback=None,
    estimation_output_callback=None, initial_state_callback=None,
    interactive=True, pad_propagation=False, validate_only=False,
):
    """Process one arc, optionally supplying campaign parameters and diagnostics.

    The optional builders receive the actual propagator/body settings and parameter
    set, respectively. With ``pad_propagation=True``, propagate an hour beyond both
    requested arc boundaries, so orbit comparisons do not use interpolator edges.
    The default remains the interactive, single-arc example.
    ``validate_only`` constructs the estimator and its partials without propagation
    or a fit, for the campaign's setup checks.
    """

    # Unpack various input arguments
    arc_index = inputs[0]

    # The requested arc bounds determine the estimation epoch and data selection.
    startDateTime = inputs[1]
    endDateTime = inputs[2]

    # Retrieve lists of relevant kernels and input files to load (TNF files, clock and orientation kernels,
    # tropospheric and ionospheric corrections)
    tnf_files = inputs[3]
    clock_files = inputs[4]
    orientation_files = inputs[5]
    tro_files = inputs[6]
    ion_files = inputs[7]
    trajectory_files = inputs[8]
    frames_def_file = inputs[9]
    structure_file = inputs[10]

    print(f"Processing arc {arc_index}: {startDateTime} to {endDateTime}")

    load_spice_kernels(
        clock_files,
        orientation_files,
        trajectory_files,
        frames_def_file,
        structure_file,
    )
    tracking_data, supplementary_data = load_tracking_data(tnf_files)

    # Define arc time interval
    arcStart = time_representation.DateTime.from_python_datetime(
        startDateTime
    ).to_epoch()
    arcEnd = time_representation.DateTime.from_python_datetime(endDateTime).to_epoch()

    time_scale_converter = time_representation.default_time_scale_converter()
    arcStart = time_scale_converter.convert_time_object(
        input_scale=time_representation.utc_scale,
        output_scale=time_representation.tdb_scale,
        input_value=time_representation.Time(arcStart),
    )
    arcEnd = time_scale_converter.convert_time_object(
        input_scale=time_representation.utc_scale,
        output_scale=time_representation.tdb_scale,
        input_value=time_representation.Time(arcEnd),
    )
    estimation_epoch = arcStart + (arcEnd - arcStart) / 2.0
    print(
        "Estimating the state at "
        f"{time_representation.DateTime.from_epoch(estimation_epoch).to_python_datetime()} TDB"
    )

    # TrackingData epochs still carry their source UTC scale. Use the nominal
    # TDB arc bounds while creating the environment; the propagation bounds are
    # refined from the converted observations below.
    environment_buffer = 7200.0 if pad_propagation else 3600.0
    environment_start_time = arcStart - environment_buffer
    environment_end_time = arcEnd + environment_buffer

    (
        bodies,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    ) = create_environment(
        environment_start_time,
        environment_end_time,
    )
    compressed_observations, obs_start_time, obs_end_time = create_observations(
        tracking_data,
        supplementary_data,
        bodies,
        arcStart,
        arcEnd,
    )
    prop_start_time = arcStart - 3600.0 if pad_propagation else obs_start_time - 3600.0
    prop_end_time = arcEnd + 3600.0 if pad_propagation else obs_end_time

    # ===================================================================================================
    # SET ANTENNA AS REFERENCE POINT FOR DOPPLER OBSERVATIONS
    print("Setting the MRO antenna reference point...")

    # Define MRO center-of-mass (COM) position w.r.t. the origin of the MRO-fixed reference frame
    com_position = [-0.001235, -1.14978, -0.001288]
    antenna_position_history = dict()

    for obs_times in compressed_observations.get_observation_times_objects():
        time = obs_times[0].to_float() - 3600.0
        while time <= obs_times[-1].to_float() + 3600.0:
            state = np.zeros((6, 1))

            # For each observation epoch, retrieve the antenna position (spice ID "-74214") w.r.t. the origin of the MRO-fixed frame (spice ID "-74000")
            state[:3, 0] = spice.get_body_cartesian_position_at_epoch(
                "-74214", "-74000", "MRO_SPACECRAFT", "none", time
            )

            # Translate the antenna position to account for the offset between the origin of the MRO-fixed frame and the COM
            state[:3, 0] = state[:3, 0] - com_position

            # Store antenna position w.r.t. COM in the MRO-fixed frame
            antenna_position_history[time] = state
            time += 60.0

    # Create tabulated ephemeris settings from antenna position history
    antenna_ephemeris_settings = environment_setup.ephemeris.tabulated(
        antenna_position_history, "-74000", "MRO_SPACECRAFT"
    )

    # Create tabulated ephemeris for the MRO antenna
    antenna_ephemeris = environment_setup.ephemeris.create_ephemeris(
        antenna_ephemeris_settings, "Antenna"
    )

    # Set the spacecraft's reference point position to that of the antenna (in the MRO-fixed frame)
    compressed_observations.set_reference_point(
        bodies,
        antenna_ephemeris,
        "Antenna",
        "MRO",
        observable_models_setup.links.LinkEndType.reflector1,
    )

    print("Setting up Doppler observation models...")

    #  Create light-time corrections list
    light_time_correction_list = list()
    light_time_correction_list.append(
        observable_models_setup.light_time_corrections.approximated_second_order_relativistic_light_time_correction(
            ["Sun"]
        )
    )

    # Add tropospheric correction
    light_time_correction_list.append(
        observable_models_setup.light_time_corrections.dsn_tabulated_tropospheric_light_time_correction(
            tro_files
        )
    )

    # Add ionospheric correction
    spacecraft_name_per_id = dict()
    spacecraft_name_per_id[74] = "MRO"
    light_time_correction_list.append(
        observable_models_setup.light_time_corrections.dsn_tabulated_ionospheric_light_time_correction(
            ion_files, spacecraft_name_per_id
        )
    )

    # Create observation model settings for the Doppler observables. This first implies creating the link ends defining all relevant
    # tracking links between various ground stations and the MRO spacecraft. The list of light-time corrections defined above is then
    # added to each of these link ends.
    doppler_link_ends = compressed_observations.link_definitions_per_observable[
        observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type
    ]

    observation_model_settings = list()
    for current_link_definition in doppler_link_ends:
        observation_model_settings.append(
            observable_models_setup.model_settings.dsn_n_way_doppler_averaged(
                current_link_definition, light_time_correction_list
            )
        )

    # Create observation simulators.
    observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        observation_model_settings, bodies
    )

    print("Computing and filtering prefit residuals...")

    # Compute and set residuals in the compressed observation collection
    observations.compute_residuals_and_dependent_variables(
        compressed_observations, observation_simulators, bodies
    )
    unfiltered_spice_residuals = np.asarray(
        compressed_observations.get_concatenated_residuals()
    ).copy()

    # Filter residuals based on the observation type
    filter_settings = {
        observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type: float(
            os.environ.get("MRO_PREFIT_RESIDUAL_CUTOFF_HZ", "0.008")
        ),
    }

    observation_filters = dict()
    for obs_type, threshold in filter_settings.items():
        parser = observations.observations_processing.observation_parser(obs_type)
        residual_filter = observations.observations_processing.observation_filter(
            observations.observations_processing.ObservationFilterType.residual_filtering,
            threshold,
        )
        observation_filters[parser] = residual_filter

    compressed_observations.filter_observations(observation_filters)
    linkEndsDict = compressed_observations.link_definition_ids

    # Initialize lists to store data from all observable types
    all_residuals = []
    all_times = []
    all_type_ids = []
    all_link_ends = []
    all_link_ids = []

    # Loop through each observable type to get its data
    for obs_type, typeName in zip(filter_settings.keys(), ["doppler"]):
        parser = observations.observations_processing.observation_parser(obs_type)

        # Get residuals, times, and link ends for the current observable type
        residuals = compressed_observations.get_concatenated_residuals(parser)
        times = compressed_observations.get_concatenated_observation_times(parser)
        link_ends_ids = compressed_observations.get_concatenated_link_definition_ids(
            parser
        )
        link_ends = [
            linkEndsDict[linkId][
                observable_models_setup.links.LinkEndType.transmitter
            ].reference_point
            + " - "
            + linkEndsDict[linkId][
                observable_models_setup.links.LinkEndType.receiver
            ].reference_point
            for linkId in link_ends_ids
        ]

        # Create a list of type identifiers
        type_ids = [typeName] * len(residuals)

        # Append the data to the main lists
        all_residuals.extend(residuals)
        all_times.extend(times)
        all_link_ends.extend(link_ends)
        all_link_ids.extend(int(link_id) for link_id in link_ends_ids)
        all_type_ids.extend(type_ids)

    # Create a single DataFrame with all the data
    residDf = pd.DataFrame(
        {
            "spice": all_residuals,
            "time": all_times,
            "link_id": all_link_ids,
            "link_ends": all_link_ends,
            "msrType": all_type_ids,
        }
    )
    residDf.attrs["unfiltered_count"] = len(unfiltered_spice_residuals)
    residDf.attrs["unfiltered_rms_hz"] = float(
        np.sqrt(np.mean(unfiltered_spice_residuals ** 2))
    )
    residDf.attrs["unfiltered_max_hz"] = float(np.max(np.abs(unfiltered_spice_residuals)))
    if prefit_callback is not None:
        prefit_callback(residDf)
    if interactive:
        plot_spice_residual_diagnostics(residDf)
        plt.show(block=False)
        plt.pause(0.1)
    if os.environ.get("MRO_PREFIT_ONLY", "0").lower() in {"1", "true", "yes"}:
        print("Prefit-only run complete; numerical propagation was not started.")
        return None

    propagator_settings, initial_state = create_propagator_settings(
        bodies,
        spacecraft_name,
        spacecraft_central_body,
        estimation_epoch,
        prop_start_time,
        prop_end_time,
    )
    # Empirical-parameter boundaries belong to the fixed physical model, not to
    # an optional perturbation of the estimated initial state.  Keep this
    # reference state separate even when no diagnostic callback is requested.
    unperturbed_initial_state = propagation.get_state_of_bodies(
        [spacecraft_name], [spacecraft_central_body], bodies, estimation_epoch
    )
    if initial_state_callback is not None:
        initial_state_callback(
            np.asarray(unperturbed_initial_state, dtype=float),
            np.asarray(initial_state, dtype=float),
            estimation_epoch.to_float(),
        )

    # =========================================================================================
    # DEFINE SET OF PARAMETERS TO BE ESTIMATED

    print("Setting up estimated parameters...")

    # Define parameters to estimate
    parameter_settings = parameters_setup.initial_states(propagator_settings, bodies)

    # Define list of additional parameters
    extra_parameters = []

    extra_parameters = [
        parameters_setup.radiation_pressure_target_direction_scaling(
            spacecraft_name, "Sun"
        ),
        # parameters_setup.radiation_pressure_target_perpendicular_direction_scaling(
        #     spacecraft_name, "Sun"
        # ),
        # parameters_setup.radiation_pressure_target_direction_scaling(
        #     spacecraft_name, "Mars"
        # ),
        # parameters_setup.radiation_pressure_target_perpendicular_direction_scaling(
        #     spacecraft_name, "Mars"
        # ),
    ]
    # Define arc start times for arc-wise empirical accelerations
    mars_gravitational_parameter = bodies.get("Mars").gravitational_parameter
    arc_start_times = empirical_arc_starts_from_reference_state(
        unperturbed_initial_state,
        mars_gravitational_parameter,
        prop_start_time.to_float(),
        prop_end_time.to_float(),
    )

    # Define empirical acceleration components to estimate for each arc
    acceleration_components_to_estimate = {
        # parameters_setup.EmpiricalAccelerationComponents.radial_empirical_acceleration_component: [
        #     parameters_setup.EmpiricalAccelerationFunctionalShapes.constant_empirical,
        #     # parameters_setup.EmpiricalAccelerationFunctionalShapes.sine_empirical,
        #     # parameters_setup.EmpiricalAccelerationFunctionalShapes.cosine_empirical,
        # ],
        parameters_setup.EmpiricalAccelerationComponents.along_track_empirical_acceleration_component: [
            parameters_setup.EmpiricalAccelerationFunctionalShapes.constant_empirical,
            parameters_setup.EmpiricalAccelerationFunctionalShapes.sine_empirical,
            parameters_setup.EmpiricalAccelerationFunctionalShapes.cosine_empirical,
        ],
        parameters_setup.EmpiricalAccelerationComponents.across_track_empirical_acceleration_component: [
            parameters_setup.EmpiricalAccelerationFunctionalShapes.constant_empirical,
            parameters_setup.EmpiricalAccelerationFunctionalShapes.sine_empirical,
            parameters_setup.EmpiricalAccelerationFunctionalShapes.cosine_empirical,
        ],
    }
    extra_parameters.append(
        parameters_setup.drag_component_scaling(spacecraft_name),
    )
    extra_parameters.append(
        parameters_setup.lift_component_scaling(spacecraft_name),
    )
    extra_parameters.append(
        parameters_setup.arcwise_empirical_accelerations(
            spacecraft_name,
            "Mars",
            acceleration_components_to_estimate,
            arc_start_times,
        )
    )

    # Add additional parameters settings
    parameter_settings += extra_parameters
    if parameter_builder is not None:
        parameter_settings = parameter_builder(propagator_settings, bodies, arc_start_times)

    # Create set of parameters to estimate
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    prior_builder = prior_builder or create_inverse_apriori_covariance
    inverse_apriori_covariance = prior_builder(
        parameters_to_estimate
    )
    nominal_parameters = parameters_to_estimate.parameter_vector.copy()

    # ==========================================================================================
    # DEFINE ESTIMATION SETTINGS AND PERFORM THE FIT

    print("Running estimation...")

    # Define estimation settings
    if os.environ.get("MRO_OBSERVATION_SIGMA_HZ") is not None:
        compressed_observations.set_constant_weight(
            float(os.environ["MRO_OBSERVATION_SIGMA_HZ"]) ** -2
        )
    estimation_input = estimation_analysis.EstimationInput(
        compressed_observations,
        inverse_apriori_covariance=inverse_apriori_covariance,
        convergence_checker=estimation_analysis.estimation_convergence_checker(
            int(os.environ.get("MRO_MAXIMUM_ITERATIONS", "5"))
        ),
        apply_apriori_parameter_deviation=os.environ.get(
            "MRO_APPLY_APRIORI_PARAMETER_DEVIATION", "1"
        ).lower()
        in {"1", "true", "yes"},
    )
    estimation_settings = dict(
        reintegrate_equations_on_first_iteration=False,
        reintegrate_variational_equations=os.environ.get(
            "MRO_REINTEGRATE_VARIATIONAL_EQUATIONS", "0"
        ).lower()
        in {"1", "true", "yes"},
        print_output_to_terminal=os.environ.get(
            "MRO_PRINT_ESTIMATION_OUTPUT", "1"
        ).lower()
        in {"1", "true", "yes"},
        save_state_history_per_iteration=True,
    )
    condition_warning_limit = os.environ.get("MRO_CONDITION_NUMBER_WARNING_LIMIT")
    if condition_warning_limit is not None:
        # Campaign workers expose every native least-squares condition number as
        # a diagnostic. Interactive use retains Tudat's defaults.
        estimation_settings.update(
            limit_condition_number_for_warning=float(condition_warning_limit),
            condition_number_warning_each_iteration=True,
        )
    estimation_input.define_estimation_settings(**estimation_settings)

    estimator = estimation_analysis.Estimator(
        bodies,
        parameters_to_estimate,
        observation_model_settings,
        propagator_settings,
        integrate_on_creation=not validate_only,
    )
    if validate_only:
        return {"parameter_count": parameters_to_estimate.parameter_set_size,
                "observations": len(residDf), "arc_bounds": (arcStart.to_float(), arcEnd.to_float()),
                "estimation_epoch": estimation_epoch.to_float()}
    estimation_output = estimator.perform_estimation(estimation_input)

    # Invoke this before checking Tudat's exception flags so a caller can retain
    # every output that the native estimator actually returned. OS termination or
    # an exception before perform_estimation returns cannot provide such output.
    if estimation_output_callback is not None:
        estimation_output_callback(
            estimation_output,
            residDf,
            np.asarray(estimation_input.weight_matrix_diagonal, dtype=float).copy(),
            np.asarray(inverse_apriori_covariance, dtype=float).copy(),
        )

    if (
        estimation_output.exception_during_propagation
        or estimation_output.exception_during_inversion
        or any(
            not iteration.dynamics_results.integration_completed_successfully
            for iteration in estimation_output.simulation_results_per_iteration
        )
    ):
        raise RuntimeError(f"MRO TNF estimation failed for arc {arc_index}")

    print("Processing estimation results...")
    bestIterIndex = estimation_output.best_iteration
    residDf["prefit"] = estimation_output.residual_history[:, 0]
    residDf["postfit"] = estimation_output.residual_history[:, bestIterIndex]

    prefit_state_history = estimation_output.simulation_results_per_iteration[
        0
    ].dynamics_results.state_history_float

    rsw_state_difference = get_rsw_state_difference(
        prefit_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    )

    prefit_rsw_df = pd.DataFrame(
        rsw_state_difference, columns=["t", "R", "T", "N", "vR", "vT", "vN"]
    )
    estimated_state_history = estimation_output.simulation_results_per_iteration[
        bestIterIndex
    ].dynamics_results.state_history_float

    rsw_state_difference = get_rsw_state_difference(
        estimated_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    )

    postfit_rsw_df = pd.DataFrame(
        rsw_state_difference, columns=["t", "R", "T", "N", "vR", "vT", "vN"]
    )

    print(f"Finished processing arc {arc_index}")
    print(
        f"Postfit Doppler residual RMS: "
        f"{np.sqrt(np.mean(np.square(residDf['postfit']))) * 1.0e3:.6f} mHz"
    )
    for component, label in zip(
        ["R", "T", "N"], ["radial", "along-track", "cross-track"]
    ):
        values = postfit_rsw_df[component]
        print(
            f"Postfit {label} position difference RMS / maximum: "
            f"{np.sqrt(np.mean(np.square(values))):.6f} / "
            f"{np.max(np.abs(values)):.6f} m"
        )

    return {
        "arc_index": arc_index,
        "residuals": residDf,
        "prefit_state_difference": prefit_rsw_df,
        "postfit_state_difference": postfit_rsw_df,
        "estimation_output": estimation_output,
        "prefit_state_history": prefit_state_history,
        "postfit_state_history": estimated_state_history,
        "nominal_parameters": nominal_parameters,
        "estimation_epoch": estimation_epoch.to_float(),
        "arc_bounds": (arcStart.to_float(), arcEnd.to_float()),
        "empirical_arc_start_times": arc_start_times,
    }


if __name__ == "__main__":
    exec_start_time = t.time()

    first_arc_start = datetime.fromisoformat("2012-01-01 03:18:01.965")
    first_arc_end = first_arc_start + timedelta(hours=12)
    print(
        "Running the recommended 12-hour MRO fit with full terminal output "
        "and interactive figures."
    )
    inputs = prepare_arc_inputs(0, first_arc_start, first_arc_end)
    arc_results = process_arc(inputs)

    if os.environ.get("MRO_PREFIT_ONLY", "0").lower() in {"1", "true", "yes"}:
        print(f"Total runtime: {t.time() - exec_start_time:.2f} s")
        plt.show()
        raise SystemExit(0)

    print("Post-processing results...")

    residDf = arc_results["residuals"].copy()
    residDf["arc_index"] = arc_results["arc_index"]
    all_residuals = [residDf]

    # Combine all residual dataframes
    if all_residuals:
        combined_residDf = pd.concat(all_residuals, ignore_index=True)
        combined_residDf = combined_residDf.sort_values(by="time")

        # Calculate overall RMS values
        prefitRMS = np.sqrt(np.mean(np.square(combined_residDf["prefit"])))
        posfitRMS = np.sqrt(np.mean(np.square(combined_residDf["postfit"])))
        spiceRMS = np.sqrt(np.mean(np.square(combined_residDf["spice"])))

        # Plot residuals
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        fig.suptitle("MRO DSN Doppler residuals")

        axes[0].set_title(
            f"Residuals with respect to reconstructed SPICE trajectory "
            f"(RMS = {spiceRMS*1e3:.2e} mHz)"
        )
        axes[0].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["spice"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        axes[1].set_title(f"Prefit residuals (RMS = {prefitRMS*1e3:.2e} mHz)")
        axes[1].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["prefit"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        axes[2].set_title(f"Postfit residuals (RMS = {(posfitRMS*1e3):.2f} mHz)")
        axes[2].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["postfit"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        for ax in axes:
            ax.set_ylabel("Doppler residual [Hz]")
            ax.grid(which="both", linestyle="--", linewidth=1.5)

        axes[0].set_ylim([-0.03, 0.03])
        axes[2].set_ylim([-0.03, 0.03])

        date_time_obj = time_representation.DateTime.from_epoch(
            time_representation.Time(combined_residDf["time"].min())
        ).to_python_datetime()
        formatted_date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
        axes[2].set_xlabel(f"Time [days since {formatted_date}]")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    else:
        print("No residual data found!")

    arc_index = str(arc_results["arc_index"])
    prefit_df = arc_results["prefit_state_difference"].copy()
    prefit_df["arc_index"] = arc_index
    postfit_df = arc_results["postfit_state_difference"].copy()
    postfit_df["arc_index"] = arc_index
    all_prefit_diff = [(arc_index, prefit_df)]
    all_postfit_diff = [(arc_index, postfit_df)]

    # Concatenate all postfit difference dataframes
    if all_postfit_diff:
        # Extract just the dataframes from the (arc_index, df) tuples
        all_postfit_dfs = [df for _, df in all_postfit_diff]

        # Concatenate into a single dataframe
        combined_postfit_df = pd.concat(all_postfit_dfs, ignore_index=True)

        # Calculate overall RMS for each component
        components = ["R", "T", "N"]
        overall_rms = {}

        for component in components:
            overall_rms[component] = np.sqrt(
                np.mean(np.square(combined_postfit_df[component]))
            )

    # Plot state differences by arc (no concatenation)
    if all_prefit_diff and all_postfit_diff:
        # Sort by arc index
        all_prefit_diff.sort(key=lambda x: int(x[0]))
        all_postfit_diff.sort(key=lambda x: int(x[0]))

        # Find global min time for consistent x-axis
        global_t_min = min([df["t"].min() for _, df in all_prefit_diff])

        # Create figure with 2x3 grid (R,T,N components for prefit and postfit)
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)

        # Components to plot
        components = ["R", "T", "N"]
        component_colors = {"R": "tab:blue", "T": "tab:orange", "N": "tab:green"}
        component_labels = {
            "R": "Radial",
            "T": "Along-track",
            "N": "Cross-track",
        }

        # Plot prefit state differences by arc (first row)
        for i, (arc_index, df) in enumerate(all_prefit_diff):
            # Get relative time in days
            time_days = (df["t"] - global_t_min) / 86400

            # Plot each component in its own panel
            for col, component in enumerate(components):
                axes[0, col].set_title(
                    f"Prefit {component_labels[component]} difference"
                )
                axes[0, col].plot(
                    time_days,
                    df[component],
                    "o-",
                    label=f"Arc {arc_index}" if i == 0 else "",
                    color=component_colors[component],
                    alpha=0.7,
                    markersize=3,
                )

                # Add light gray vertical lines to separate arcs
                if i < len(all_prefit_diff) - 1:
                    arc_end = time_days.max()
                    axes[0, col].axvline(
                        arc_end, color="gray", linestyle="-", linewidth=0.5, alpha=0.3
                    )

        # Plot postfit state differences by arc (second row)
        for i, (arc_index, df) in enumerate(all_postfit_diff):
            # Get relative time in days
            time_days = (df["t"] - global_t_min) / 86400

            # Plot each component in its own panel
            for col, component in enumerate(components):
                axes[1, col].set_title(
                    f"Postfit {component_labels[component]} difference "
                    f"(RMS = {overall_rms[component]:.2f} m)"
                )
                axes[1, col].plot(
                    time_days,
                    df[component],
                    "o-",
                    label=f"Arc {arc_index}" if i == 0 else "",
                    color=component_colors[component],
                    alpha=0.7,
                    markersize=3,
                )

                # Add light gray vertical lines to separate arcs
                if i < len(all_postfit_diff) - 1:
                    arc_end = time_days.max()
                    axes[1, col].axvline(
                        arc_end, color="gray", linestyle="-", linewidth=0.5, alpha=0.3
                    )

        # Set titles and labels
        for col, component in enumerate(components):
            # Add y-axis labels
            axes[0, col].set_ylabel(f"{component_labels[component]} difference [m]")
            axes[1, col].set_ylabel(f"{component_labels[component]} difference [m]")

            # Add grid to all subplots
            axes[0, col].grid(which="both", linestyle="--", linewidth=1.5)
            axes[1, col].grid(which="both", linestyle="--", linewidth=1.5)

        date_time_obj = time_representation.DateTime.from_epoch(
            time_representation.Time(global_t_min)
        ).to_python_datetime()
        formatted_date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")

        # Add x-axis labels to bottom row
        for col in range(3):
            axes[1, col].set_xlabel(f"Time [days since {formatted_date}]")

        fig.suptitle(
            "MRO trajectory differences with respect to reconstructed SPICE trajectory"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    print(f"Total runtime: {t.time() - exec_start_time:.2f} s")
    plt.show()
