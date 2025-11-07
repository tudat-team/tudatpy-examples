# %%
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from matplotlib import pyplot as plt


from mro_utils import get_mro_files, macromodel_mro, get_rsw_state_difference

from tudatpy.util import redirect_std
from tudatpy.interface import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.data import processTrk234

from tudatpy.dynamics import (
    environment_setup,
    propagation_setup,
    parameters_setup,
    propagation,
    parameters,
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


def process_arc(inputs):
    """Process a single arc and save results"""

    # Unpack various input arguments
    arc_index = inputs[0]

    # Convert start and end datetime objects to Tudat Time variables. A time buffer of one day is subtracted/added to the start/end date
    # to ensure that the simulation environment covers the full time span of the loaded TNF files. This is mostly needed because some TNF
    # files - while typically assigned to a certain date - actually spans over (slightly) longer than one day. Without this time buffer,
    # some observation epochs might thus lie outside the time boundaries within which the dynamical environment is defined.
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

    # Create output folder for this specific arc
    arc_output_folder = f"{output_folder_base}/arc_{arc_index}/"
    if not os.path.exists(arc_output_folder):
        os.makedirs(arc_output_folder)

    # Create log file for this arc
    arc_log_file = arc_output_folder + f"fitLog.txt"
    with open(arc_log_file, "w") as f:
        f.write(f"Arc: {arc_index}\n")

    print(f"Processing arc {arc_index}: {startDateTime} to {endDateTime}")

    spice.load_standard_kernels()

    # Load MRO orientation kernels (over the entire relevant time period).
    for orientation_file in orientation_files:
        spice.load_kernel(orientation_file)

    # Load MRO clock files
    for clock_file in clock_files:
        spice.load_kernel(clock_file)

    # Load MRO frame definition file (useful for HGA and spacecraft-fixed frames definition)
    spice.load_kernel(frames_def_file)

    # Load MRO trajectory kernels
    for trajectory_file in trajectory_files:
        spice.load_kernel(trajectory_file)

    # Load MRO spacecraft structure file (for antenna position in spacecraft-fixed frame)
    spice.load_kernel(structure_file)

    # Remove first TNF file to avoid issues with time coverage
    tnf_files = tnf_files[1:]

    # Data for arc 4 is in the previous day TNF file
    if arc_index == 4:
        tnf_files.append("mro_kernels/mromagr2012_016_0520xmmmv1.tnf")

    # LOAD TNF OBSERVATIONS AND PERFORM PRE-PROCESSING STEPS
    tnfProcessor = processTrk234.Trk234Processor(
        tnf_files,
        ["doppler"],
        spacecraft_name="MRO",
    )
    original_observations = tnfProcessor.process()

    # Remove observation outside the arc time interval
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

    # Filter observations to the arc time interval
    arc_filter = observations.observations_processing.observation_filter(
        observations.observations_processing.ObservationFilterType.time_bounds_filtering,
        arcStart.to_float(),
        arcEnd.to_float(),
        use_opposite_condition=True,
    )
    original_observations.filter_observations(arc_filter)
    original_observations.remove_empty_observation_sets()

    # Compress Doppler observations from 1.0 s integration time to 60.0 s
    compressed_observations = (
        observations_setup.observations_wrapper.create_compressed_doppler_collection(
            original_observations, 60, 10
        )
    )

    # Add transpondr delay
    compressed_observations.set_transponder_delay("MRO", 1.4149e-6)

    # Buffer model/propagation start and end times
    observation_time_limits = original_observations.time_bounds_time_object
    obs_start_time = observation_time_limits[0]
    obs_end_time = observation_time_limits[1]

    prop_start_time = observation_time_limits[0] - 3600.0
    prop_end_time = observation_time_limits[1] + 3600.0

    # ====================
    # Create default body settings for celestial bodies
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
        prop_start_time.to_float(),
        prop_end_time.to_float(),
        global_frame_origin,
        global_frame_orientation,
    )

    # ====================
    # Earth
    # Modify default shape, rotation, and gravity field settings for the Earth
    body_settings.get("Earth").shape_settings = (
        environment_setup.shape.oblate_spherical_spice()
    )
    body_settings.get("Earth").rotation_model_settings = (
        environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006,
            global_frame_orientation,
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                prop_start_time.to_float(),
                prop_end_time.to_float(),
                3600.0,
            ),
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                prop_start_time.to_float(),
                prop_end_time.to_float(),
                3600.0,
            ),
            interpolators.interpolator_generation_settings(
                interpolators.cubic_spline_interpolation(),
                prop_start_time.to_float(),
                prop_end_time.to_float(),
                60.0,
            ),
        )
    )
    body_settings.get("Earth").gravity_field_settings.associated_reference_frame = (
        "ITRS"
    )

    # Set up DSN ground stations
    body_settings.get("Earth").ground_station_settings = (
        environment_setup.ground_station.dsn_stations()
    )

    # ====================
    # Mars
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

    # Define gravity field variations for the tides on Mars
    body_settings.get("Mars").gravity_field_variation_settings = [
        environment_setup.gravity_field_variation.solid_body_tide("Sun", 0.1697, 2),
        environment_setup.gravity_field_variation.solid_body_tide("Phobos", 0.1697, 2),
    ]

    # Define Mars atmosphere settings
    body_settings.get("Mars").atmosphere_settings = (
        environment_setup.atmosphere.mars_dtm()
    )

    # Define Mars irradiance-based radiation pressure settings
    luminosity_settings = (
        environment_setup.radiation_pressure.irradiance_based_constant_luminosity(
            250, 3.4e6
        )
    )
    body_settings.get("Mars").radiation_source_settings = (
        environment_setup.radiation_pressure.isotropic_radiation_source(
            luminosity_settings
        )
    )

    # MRO
    spacecraft_name = "MRO"
    spacecraft_central_body = "Mars"
    body_settings.add_empty_settings(spacecraft_name)

    # Retrieve translational ephemeris from SPICE
    body_settings.get(spacecraft_name).ephemeris_settings = (
        environment_setup.ephemeris.interpolated_spice(
            prop_start_time.to_float(),
            prop_end_time.to_float(),
            10.0,
            spacecraft_central_body,
            global_frame_orientation,
        )
    )

    # Retrieve rotational ephemeris from SPICE
    body_settings.get(spacecraft_name).rotation_model_settings = (
        environment_setup.rotation_model.spice(
            global_frame_orientation, spacecraft_name + "_SPACECRAFT", ""
        )
    )

    body_settings.get(spacecraft_name).constant_mass = 1262.39  # [kg]
    body_settings.get(spacecraft_name).vehicle_shape_settings = macromodel_mro()

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Set MRO aerodynamics settings using spacecraft macromodel, variable cross-section and constant coefficients
    drag_coefficient = 2.0  # from Mazarico et al.
    lift_coefficient = 0.01  # set different to zero to estimate lift
    # lift_coefficient = 0  # set different to zero to estimate lift
    ssh = 0
    aero_coefficient_settings = (
        environment_setup.aerodynamic_coefficients.constant_variable_cross_section(
            [drag_coefficient, 0, lift_coefficient], ssh
        )
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, spacecraft_name, aero_coefficient_settings
    )

    # Set MRO radiation pressure settings using spacecraft macromodel and variable cross-section
    ssh = 100
    occulting_bodies_dict = dict(Sun=["Mars"])
    pixel_source_dict = dict(Sun=ssh)
    radiation_pressure_settings = (
        environment_setup.radiation_pressure.panelled_radiation_target(
            occulting_bodies_dict, pixel_source_dict
        )
    )
    environment_setup.add_radiation_pressure_target_model(
        bodies, spacecraft_name, radiation_pressure_settings
    )

    tnfProcessor.set_tnf_information_in_bodies(bodies)

    # ===================================================================================================
    # SET ANTENNA AS REFERENCE POINT FOR DOPPLER OBSERVATIONS

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

    # Compute and set residuals in the compressed observation collection
    observations.compute_residuals_and_dependent_variables(
        compressed_observations, observation_simulators, bodies
    )

    # Filter residuals based on the observation type
    filter_settings = {
        observable_models_setup.model_settings.dsn_n_way_averaged_doppler_type: 0.1,
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
        all_type_ids.extend(type_ids)

    # Create a single DataFrame with all the data
    residDf = pd.DataFrame(
        {
            "spice": all_residuals,
            "time": all_times,
            "link_ends": all_link_ends,
            "msrType": all_type_ids,
        }
    )

    # =========================================================================================
    # DEFINE PROPAGATION SETTINGS

    # Define list of accelerations acting on GRAIL
    accelerations_settings_spacecraft = dict(
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.paneled_target
            ),
        ],
        Mars=[
            propagation_setup.acceleration.spherical_harmonic_gravity(120, 120),
            propagation_setup.acceleration.aerodynamic(),
            propagation_setup.acceleration.radiation_pressure(
                environment_setup.radiation_pressure.paneled_target
            ),
            propagation_setup.acceleration.empirical(),
        ],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Phobos=[propagation_setup.acceleration.point_mass_gravity()],
        Deimos=[propagation_setup.acceleration.point_mass_gravity()],
    )

    # Create accelerations settings dictionary
    acceleration_settings = {spacecraft_name: accelerations_settings_spacecraft}

    # Create acceleration models from settings
    bodies_to_propagate = [spacecraft_name]
    central_bodies = [spacecraft_central_body]
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Define integrator settings
    integration_step = 30.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        time_representation.Time(0, integration_step),
        propagation_setup.integrator.rkf_78,
    )

    # Retrieve initial state from SPICE
    initial_state = propagation.get_state_of_bodies(
        bodies_to_propagate, central_bodies, bodies, prop_start_time
    )

    # Define propagator settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        prop_start_time,
        integrator_settings,
        propagation_setup.propagator.time_termination(obs_end_time.to_float()),
    )

    # =========================================================================================
    # DEFINE SET OF PARAMETERS TO BE ESTIMATED

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
    keplerian_state = element_conversion.cartesian_to_keplerian(
        initial_state, mars_gravitational_parameter
    )
    semi_major_axis = keplerian_state[0]
    orbital_period = (
        2.0 * np.pi * np.sqrt(semi_major_axis**3 / mars_gravitational_parameter)
    )

    # Save arc start times
    arc_start_times = []
    current_arc_start_time = obs_start_time.to_float()
    while current_arc_start_time < obs_end_time.to_float():
        arc_start_times.append(current_arc_start_time)
        current_arc_start_time += orbital_period

    with open(arc_output_folder + f"arc_start_times.pkl", "wb") as f:
        pickle.dump(arc_start_times, f)

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
        parameters_setup.arcwise_empirical_accelerations(
            spacecraft_name,
            "Mars",
            acceleration_components_to_estimate,
            arc_start_times,
        )
    )

    extra_parameters.append(
        parameters_setup.drag_component_scaling(spacecraft_name),
    )
    extra_parameters.append(
        parameters_setup.lift_component_scaling(spacecraft_name),
    )

    # Add additional parameters settings
    parameter_settings += extra_parameters

    # Create set of parameters to estimate
    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )

    nominal_parameters = parameters_to_estimate.parameter_vector

    # define a priori
    posSigma = [1e3] * 3  # m
    velSigma = [1e-1] * 3  # m/s
    srpScaleSigma = [2]  # dimensionless
    dragSigma = [2]  # dimensionless
    liftSigma = [2]  # dimensionless
    # liftSigma = []  # dimensionless
    # radialAcc = [1e-6] * 1 * len(arc_start_times)  # m/s^2
    radialAcc = []  # m/s^2
    alongAcc = [1e-6] * 3 * len(arc_start_times)  # m/s^2
    acrossAcc = [1e-6] * 3 * len(arc_start_times)  # m/s^2
    # alongAcc = []  # m/s^2
    # acrossAcc = []  # m/s^2
    all_sigmas = (
        posSigma
        + velSigma
        + srpScaleSigma
        + dragSigma
        + liftSigma
        + radialAcc
        + alongAcc
        + acrossAcc
    )

    # Create the a priori covariance matrix (diagonal) from the sigmas.
    apriori_covariance = np.diag(1 / np.square(all_sigmas))

    # ==========================================================================================
    # DEFINE ESTIMATION SETTINGS AND PERFORM THE FIT

    # Create estimator
    estimator = estimation_analysis.Estimator(
        bodies, parameters_to_estimate, observation_model_settings, propagator_settings
    )

    # Define estimation settings
    estimation_input = estimation_analysis.EstimationInput(
        compressed_observations,
        inverse_apriori_covariance=apriori_covariance,
        convergence_checker=estimation_analysis.estimation_convergence_checker(6),
    )
    estimation_input.define_estimation_settings(
        reintegrate_equations_on_first_iteration=False,
        reintegrate_variational_equations=True,
        print_output_to_terminal=True,
        save_state_history_per_iteration=True,
    )

    # Perform estimation
    with redirect_std(arc_log_file, True, True):
        print(f"Arc {arc_index} interval:")
        print(
            time_representation.DateTime.from_epoch(arcStart),
            time_representation.DateTime.from_epoch(arcEnd),
        )
        print(
            "Observation time bounds:",
            time_representation.DateTime.from_epoch(obs_start_time),
            " - ",
            time_representation.DateTime.from_epoch(obs_end_time),
        )

        print(f"Propagation time limits:")
        print(
            time_representation.DateTime.from_epoch(prop_start_time),
            time_representation.DateTime.from_epoch(prop_end_time),
        )
        parameters.print_parameter_names(parameters_to_estimate)
        print(
            "Number of parameters to estimate:",
            len(nominal_parameters),
        )
        print("Nominal values +- apriori:")
        for value, sigma in zip(nominal_parameters, all_sigmas):
            print(f"{value:.6e} +- {sigma:.6e}")
        print("\n")
        estimation_output = estimator.perform_estimation(estimation_input)
        print("\n")
        print("Corrected values +- uncertainty:")
        for value, sigma in zip(
            estimation_output.final_parameters, estimation_output.formal_errors
        ):
            print(f"{value:.6e} +- {sigma:.6e}")

    outputDict = {}
    outputDict["nominal"] = nominal_parameters
    outputDict["correted"] = estimation_output.final_parameters
    outputDict["formal_errors"] = estimation_output.formal_errors

    with open(arc_output_folder + f"estimationOutputDict.pkl", "wb") as f:
        pickle.dump(outputDict, f)

    # Collect results
    bestIterIndex = estimation_output.best_iteration
    residDf["prefit"] = estimation_output.residual_history[:, 0]
    residDf["postfit"] = estimation_output.residual_history[:, bestIterIndex]
    residDf.to_pickle(arc_output_folder + f"residDf.pkl")

    prefit_state_history = estimation_output.simulation_results_per_iteration[
        0
    ].dynamics_results.state_history_float

    rsw_state_difference = get_rsw_state_difference(
        prefit_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    )

    rsw_df = pd.DataFrame(
        rsw_state_difference, columns=["t", "R", "T", "N", "vR", "vT", "vN"]
    )
    rsw_df.to_pickle(arc_output_folder + f"rsw_state_difference_prefit.pkl")
    estimated_state_history = estimation_output.simulation_results_per_iteration[
        bestIterIndex
    ].dynamics_results.state_history_float

    rsw_state_difference = get_rsw_state_difference(
        estimated_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation,
    )

    rsw_df = pd.DataFrame(
        rsw_state_difference, columns=["t", "R", "T", "N", "vR", "vT", "vN"]
    )
    rsw_df.to_pickle(arc_output_folder + f"rsw_state_difference_postfit.pkl")

    print(f"Finished processing arc {arc_index}")

    return arc_index


if __name__ == "__main__":
    exec_start_time = t.time()

    # Set up the output folder
    output_folder_base = "mro_outputs"
    if not os.path.exists(output_folder_base):
        os.makedirs(output_folder_base)

    arcs = [
        (
            datetime.fromisoformat("2012-01-01 03:18:01.965"),
            datetime.fromisoformat("2012-01-04 01:58:15.132"),
        ),
        (
            datetime.fromisoformat("2012-01-04 02:25:09.706"),
            datetime.fromisoformat("2012-01-07 02:55:23.122"),
        ),
        (
            datetime.fromisoformat("2012-01-07 03:23:14.407"),
            datetime.fromisoformat("2012-01-10 02:03:27.113"),
        ),
        (
            datetime.fromisoformat("2012-01-10 02:22:44.539"),
            datetime.fromisoformat("2012-01-13 02:52:58.104"),
        ),
        (
            datetime.fromisoformat("2012-01-13 03:15:38.112"),
            datetime.fromisoformat("2012-01-16 02:05:51.095"),
        ),
        (
            datetime.fromisoformat("2012-01-16 02:22:44.352"),
            datetime.fromisoformat("2012-01-19 02:52:57.085"),
        ),
        (
            datetime.fromisoformat("2012-01-19 03:17:17.831"),
            datetime.fromisoformat("2012-01-22 01:57:31.076"),
        ),
    ]

    # Set up multiprocessing pool
    num_processes = len(arcs)

    inputs = []
    for i, arc in enumerate(arcs):

        startEpoch = arc[0]
        endEpoch = arc[1]
        startEpochWithBuffer = startEpoch - timedelta(days=1)
        endEpochWithBuffer = endEpoch + timedelta(days=1)

        print("Files for arc {}".format(i))
        # First retrieve the names of all the relevant kernels and data files necessary to cover the specified time interval
        (
            clock_files,
            orientation_files,
            tro_files,
            ion_files,
            tnf_files,
            trajectory_files,
            frames_def_file,
            structure_file,
        ) = get_mro_files("mro_kernels/", startEpochWithBuffer, endEpochWithBuffer)
        print("\n")

        # Construct a list of input arguments containing the arguments needed this specific parallel run.
        # These include the start and end dates, along with the names of all relevant kernels and data files that should be loaded
        inputs.append(
            [
                i,
                startEpoch,
                endEpoch,
                tnf_files,
                clock_files,
                orientation_files,
                tro_files,
                ion_files,
                trajectory_files,
                frames_def_file,
                structure_file,
            ]
        )

    # Process arcs in parallel using enumerate to get index
    with multiprocessing.get_context("fork").Pool(num_processes) as pool:
        pool.map(process_arc, inputs)

    arcDirs = [
        os.path.join(output_folder_base, d)
        for d in os.listdir(output_folder_base)
        if os.path.isdir(os.path.join(output_folder_base, d))
    ]

    # Plot residuals from all arcs
    all_residuals = []
    all_arc_times = []
    for arc_dir in arcDirs:
        arc_index = os.path.basename(arc_dir).split("_")[-1]
        print(f"Processing residuals from {arc_dir}")

        try:
            # Load residuals
            residDf = pd.read_pickle(f"{arc_dir}/residDf.pkl")
            residDf["arc_index"] = arc_index  # Add identifier column
            all_residuals.append(residDf)

            # Load arc times
            arc_times = pickle.load(open(f"{arc_dir}/arc_start_times.pkl", "rb"))
            for time in arc_times:
                all_arc_times.append(time)
        except FileNotFoundError as e:
            print(f"Warning: Could not load data from {arc_dir}: {e}")

    # Combine all residual dataframes
    if all_residuals:
        combined_residDf = pd.concat(all_residuals, ignore_index=True)
        combined_residDf = combined_residDf.sort_values(by="time")
        all_arc_times = np.array(sorted(all_arc_times))

        # Calculate overall RMS values
        prefitRMS = np.sqrt(np.mean(np.square(combined_residDf["prefit"])))
        posfitRMS = np.sqrt(np.mean(np.square(combined_residDf["postfit"])))
        spiceRMS = np.sqrt(np.mean(np.square(combined_residDf["spice"])))

        # Plot residuals
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        axes[0].set_title(f"w.r.t. SPK, RMS = {spiceRMS*1e3:.2e} mHz")
        axes[0].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["spice"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        axes[1].set_title(f"Prefit RMS = {prefitRMS*1e3:.2e} mHz")
        axes[1].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["prefit"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        axes[2].set_title(f"Postfit RMS = {(posfitRMS*1e3):.2f} mHz")
        axes[2].scatter(
            (combined_residDf["time"] - combined_residDf["time"].min()) / 86400,
            combined_residDf["postfit"],
            s=20,
            marker="o",
            alpha=0.7,
        )

        for ax in axes:
            ax.set_ylabel("Residuals [Hz]")
            ax.grid(which="both", linestyle="--", linewidth=1.5)

        axes[0].set_ylim([-0.03, 0.03])
        axes[2].set_ylim([-0.03, 0.03])

        date_time_obj = time_representation.DateTime.from_epoch(
            time_representation.Time(combined_residDf["time"].min())
        ).to_python_datetime()
        formatted_date = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
        axes[2].set_xlabel(f"Time [days since {formatted_date}]")

    else:
        print("No residual data found!")

    # Collect all state difference data
    all_prefit_diff = []
    all_postfit_diff = []

    for arc_dir in arcDirs:
        arc_index = os.path.basename(arc_dir).split("_")[1]
        print(f"Processing state differences from {arc_dir}")

        if int(arc_index) > 4:
            continue

        try:
            # Load prefit state differences
            prefit_df = pd.read_pickle(f"{arc_dir}/rsw_state_difference_prefit.pkl")
            prefit_df["arc_index"] = arc_index
            all_prefit_diff.append((arc_index, prefit_df))

            # Load postfit state differences
            postfit_df = pd.read_pickle(f"{arc_dir}/rsw_state_difference_postfit.pkl")
            postfit_df["arc_index"] = arc_index
            all_postfit_diff.append((arc_index, postfit_df))
        except FileNotFoundError as e:
            print(f"Warning: Could not load state difference data from {arc_dir}: {e}")

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

        # Plot prefit state differences by arc (first row)
        for i, (arc_index, df) in enumerate(all_prefit_diff):
            # Get relative time in days
            time_days = (df["t"] - global_t_min) / 86400

            # Plot each component in its own panel
            for col, component in enumerate(components):
                # axes[0, col].set_title(f"Prefit RMS = {overall_rms[component]:.2f} m")
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
                axes[1, col].set_title(f"Postfit RMS = {overall_rms[component]:.2f} m")
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
            axes[0, col].set_ylabel(f"{component} Difference [m]")
            axes[1, col].set_ylabel(f"{component} Difference [m]")

            # Add grid to all subplots
            axes[0, col].grid(which="both", linestyle="--", linewidth=1.5)
            axes[1, col].grid(which="both", linestyle="--", linewidth=1.5)

        # Add x-axis labels to bottom row
        for col in range(3):
            axes[1, col].set_xlabel("Time [days]")

        plt.show()
