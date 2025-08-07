# Tudat imports for propagation and estimation
from mypy.checkexpr import defaultdict
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, simulator
from tudatpy.estimation import observable_models_setup,observable_models, observations_setup, observations, estimation_analysis
from tudatpy.astro import element_conversion, time_representation
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tudatpy.data.spacetrack import SpaceTrackQuery
from tudatpy.dynamics import parameters_setup, parameters, propagation, propagation_setup
from tudatpy.dynamics.environment_setup.ephemeris import create_ephemeris_settings
from tudatpy.dynamics.propagation_setup.acceleration import orbital_regimes


# Load spice standard kernels
spice.load_standard_kernels()

norad_id = str(64865)
username = 'l.gisolfi@tudelft.nl'
password = 'l.gisolfi*tudelft.nl'

# Initialize SpaceTrackQuery
spactrack_request = SpaceTrackQuery(username, password)
tle_query = spactrack_request.DownloadTle(spactrack_request)
omm_utils = spactrack_request.OMMUtils(tle_query)
AccelerationSettingsPerRegime = orbital_regimes.GetAccelerationSettingsPerRegime()
# OMM Dict
json_dict = tle_query.single_norad_id( norad_id)

# Retrieve TLEs
tle_dict = omm_utils.get_tles(json_dict)
tle_line1, tle_line2 = tle_dict[norad_id][0], tle_dict[norad_id][1]

# Retrieve TLE Reference epoch, this will be start epoch of simulation
tle_reference_epoch = omm_utils.get_tle_reference_epoch(tle_line1)

number_of_pod_iterations = 6 # number of iterations for our estimation
timestep_global = 5 # timestep of 120 seconds for our estimation
time_buffer = 60 # uncomment only if needed.

# Define Simulation Start and End (Date)Times
observations_start = tle_reference_epoch # set as tle reference epoch for now, but can vary
observations_end = tle_reference_epoch + timedelta(seconds=60*30) # one hour after observation start
observations_start_epoch = time_representation.datetime_to_tudat(observations_start).epoch()
observations_end_epoch = time_representation.datetime_to_tudat(observations_end).epoch()

# define the frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# List the bodies for our environment
bodies_to_create = [
    "Sun",
    "Earth",
    "Moon",
]

CreateEphemerisSettings = create_ephemeris_settings.CreateEphemerisSettings(spactrack_request, AccelerationSettingsPerRegime)

orbital_regime, orbital_regime_definition = omm_utils.get_orbital_regime(json_dict[0])
print(f'Object (NORAD ID): {norad_id}.\nOrbital Regime: {orbital_regime}, Regime Definition: {orbital_regime_definition}\n')

compare_dynamical_model_vs_sgp4 =True

if compare_dynamical_model_vs_sgp4:
    tabulated_ephemeris = dict()
    bodies = dict()
    integrator_settings = dict()
    acceleration_settings = dict()
    propagator_settings = dict()
    body_settings = dict()
    for dynamical_model in [orbital_regime, 'SGP4']:
        # !!!!!!!!!!!!!!!!!
        # The following function in CreateEphemerisSettings should be modified from source to handle non-starlink, non cubesat satellites
        # (change default drag, srp coeff, mass, etc...)
        # !!!!!!!!!!!!!!!!!
        if dynamical_model != 'SGP4':
            tabulated_ephemeris[dynamical_model], bodies[dynamical_model],\
            integrator_settings[dynamical_model], acceleration_settings[dynamical_model],\
            propagator_settings[dynamical_model], body_settings[dynamical_model] = CreateEphemerisSettings.set_object_ephemeris_settings(
            object_name = norad_id,
            tle_line_1 = tle_line1,
            tle_line_2 = tle_line2,
            bodies_to_create = bodies_to_create,
            frame_origin = global_frame_origin,
            frame_orientation = global_frame_orientation,
            orbital_regime = dynamical_model,
            aerodynamic = True,
            srp = True,
            mass = 260,
            reference_area = 20,
            drag_coefficient = 1.2,
            radiation_coefficient = 1.2
            )

        else:
            tabulated_ephemeris[dynamical_model], bodies[dynamical_model] = CreateEphemerisSettings.set_object_ephemeris_settings(
                object_name = norad_id,
                tle_line_1 = tle_line1,
                tle_line_2 = tle_line2,
                bodies_to_create = bodies_to_create,
                frame_origin = global_frame_origin,
                frame_orientation = global_frame_orientation,
                sgp4_flag= True,
                aerodynamic = True,
                srp = True,
                mass = 260,
                reference_area = 20,
                drag_coefficient = 1.2,
                radiation_coefficient = 1.2
            )

        # Validation print statement: check that aerodynamic coefficients were set
        print(bodies[dynamical_model].get(norad_id).aerodynamic_coefficient_interface.current_coefficients)
        print(bodies[dynamical_model].get('Earth').atmosphere_model)

        if not propagator_settings[dynamical_model]:
            # !!!!!!!!!!!!!!!!!!!
            # temporary solution until when we define the proper propagator settings for sgp4.
            # !!!!!!!!!!!!!!!!!!!
            propagator_settings[dynamical_model] = propagator_settings[orbital_regime]

        initial_state = bodies[dynamical_model].get(norad_id).ephemeris.cartesian_state(observations_start_epoch)

        # Setup parameters settings to propagate the state transition matrix
        parameter_settings = parameters_setup.initial_states(
            propagator_settings[dynamical_model], bodies[dynamical_model]
        )
        parameters_to_estimate = parameters_setup.create_parameter_set(
            parameter_settings, bodies[dynamical_model], propagator_settings[dynamical_model]
        )

        # Save the true initial state to later analyse the error
        truth_parameters = parameters_to_estimate.parameter_vector
        print(f'Truth Parameters: \n {truth_parameters}')

        # Create light-time corrections list
        light_time_correction_list = list()
        light_time_correction_list.append(
            observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]))

        # Define TU Delft Rooftop station and coordinates
        reference_point = "TU_DELFT_ROOFTOP"
        reference_point_backup = "ARUBA_ROOFTOP"
        stations_altitude = 0.0
        delft_latitude = np.deg2rad(52.00667)
        delft_longitude = np.deg2rad(4.35556)

        # Uncomment to add link end in Aruba
        #aruba_latitude = np.deg2rad(12.517572)
        #aruba_longitude = np.deg2rad(-69.9649462)

        environment_setup.add_ground_station(
            bodies[dynamical_model].get_body("Earth"),
            reference_point,
            [stations_altitude, delft_latitude, delft_longitude],
            element_conversion.geodetic_position_type)

        # Uncomment to add link end in Aruba
        #environment_setup.add_ground_station(
        #    bodies.get_body("Earth"),
        #    reference_point_backup,
        #    [stations_altitude, aruba_latitude, aruba_longitude],
        #    element_conversion.geodetic_position_type)

        link_ends_list = [{observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', reference_point),
                           observable_models_setup.links.transmitter: observable_models_setup.links.body_origin_link_end_id(norad_id)}]

        # Add this to the list if probing with link end in Aruba
        #{observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', reference_point_backup),
        # observable_models_setup.links.transmitter: observable_models_setup.links.body_origin_link_end_id(norad_id)}, ]

        # Create Link Ends
        link_definition_list = []
        for link_ends in link_ends_list:
            link_definition_list.append(observable_models_setup.links.link_definition(link_ends))
        observation_model_settings = list()

        # Create observation model settings
        for link_definition in link_definition_list:
            observation_model_settings.append(observable_models_setup.model_settings.angular_position(
                link_definition, light_time_correction_list))

        # Create observations times. Results of the estimation slightly depend on this time buffer.
        # Consider exploring why things do not work with time_buffer = 0
        observation_times = np.arange(observations_start_epoch + time_buffer, observations_end_epoch - time_buffer, timestep_global)
        observation_simulation_settings = list()
        body_state_history = dict()

        # Create Simulation (propagation) Settings
        for link_definition in link_definition_list:
            observation_simulation_settings.append(observations_setup.observations_simulation_settings.tabulated_simulation_settings(
                observable_models_setup.model_settings.angular_position_type,
                link_definition,
                observation_times))

        # Add random noise to the observations
        noise_level = 1e-6 # typical astrometry error for detection is around 1 arcsec, corresponding to 1e-6 radians.
        observations_setup.random_noise.add_gaussian_noise_to_observable(
            observation_simulation_settings,
            noise_level,
            observable_models_setup.model_settings.angular_position_type
        )

        # Create observation simulators
        observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(observation_model_settings, bodies[dynamical_model])

        # Retrieve observation collection
        norad_id_simulated_observations = observations_setup.observations_wrapper.simulate_observations(
            observation_simulation_settings,
            observation_simulators,
            bodies[dynamical_model])

        # Add weights to the observation collection
        # (if not set, default = 1 [radians]. This would be too big!)
        # When using radio links, default = 1 [m]. In that case, that would be fine...

        norad_id_simulated_observations.set_constant_weight(noise_level)

        # Compute and set residuals in the compressed observation collection
        observations.compute_residuals_and_dependent_variables(norad_id_simulated_observations, observation_simulators, bodies[dynamical_model])

        # Filter residual outliers (if any)
        norad_id_simulated_observations.filter_observations(
            observations.observations_processing.observation_filter(observations.observations_processing.residual_filtering, 0.1))

        # Retrieve relevant quantities (observations, residuals, observation times) from observation collection
        concatenated_observations = norad_id_simulated_observations.concatenated_observations
        concatenated_residuals = norad_id_simulated_observations.get_concatenated_residuals()
        concatenated_times= norad_id_simulated_observations.concatenated_times

        # Separate Ra and Dec coordinates (they are found together in concatenated observations)
        concatenated_ra = concatenated_observations[::2]
        concatenated_dec = concatenated_observations[1::2]

        # Convert to degrees
        concatenated_ra_deg = [np.rad2deg(ra) for ra in concatenated_ra]
        concatenated_dec_deg = [np.rad2deg(dec) for dec in concatenated_dec]

        # Plot observations
        fig, axs = plt.subplots(
            1,
            1,
            figsize=(9, 7),
            sharex=True,
            sharey=False,
        )

        # plot the residuals, split between RA and DEC types
        for idx, ax in enumerate(fig.get_axes()):
            ax.grid()
            ax.scatter(
                concatenated_times[::2],
                concatenated_ra_deg,
                marker="+",
                s=40,
                label="Right Ascension",
            )

            ax.scatter(
                concatenated_times[::2],
                concatenated_dec_deg,
                marker="+",
                s=40,
                label="Declination",
            )

        plt.legend()
        plt.tight_layout()
        plt.xlabel('Observation Times [s]')
        plt.ylabel('Ra and Dec [deg]')
        plt.show()

        # Plot PRE-FIT Residuals
        plt.scatter(concatenated_times,concatenated_residuals, s = 8)
        plt.xlabel('Observation Times [s]')
        plt.ylabel('Pre-Fit Residuals [rad]')
        plt.show()


        # Set up the estimator
        estimator = estimation_analysis.Estimator(
            bodies=bodies[dynamical_model],
            estimated_parameters=parameters_to_estimate,
            observation_settings=observation_model_settings,
            propagator_settings=propagator_settings[dynamical_model],
        )

        # provide the observation collection as input, and limit number of iterations for estimation.
        estimation_input = estimation_analysis.EstimationInput(
            observations_and_times=norad_id_simulated_observations,
            convergence_checker=estimation_analysis.estimation_convergence_checker(
                maximum_iterations=10,
            ),
        )

        # Set estimation methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration = True, reintegrate_variational_equations=True)


        # Perform the estimation
        estimation_output = estimator.perform_estimation(estimation_input)

        # retrieve the estimated initial state and compare it to the truth.

        initial_state_updated = parameters_to_estimate.parameter_vector
        print('Done with the estimation...\n')
        print(f'Updated initial states: {initial_state_updated}\n')

        vector_error_final = (np.array(initial_state_updated) - truth_parameters)[0:3]
        error_magnitude_final = np.linalg.norm(vector_error_final)/1000

        print(
            f"{norad_id} final error to TLE initial state: {round(error_magnitude_final, 2)} km\n"
        )


        # These three lines might be useful later when adding multiple parameters to estimate
        simulator_object = estimation_output.simulation_results_per_iteration[-1]
        state_history = simulator_object.dynamics_results.state_history
        dependent_variable_history = simulator_object.dynamics_results.dependent_variable_history


        # Plot the propagated orbit and highlight truth vs estimated state
        ephemeris_state = list()
        for epoch in state_history.keys():
            ephemeris_state.append(state_history[epoch])

        # Extract x, y, z components
        x_values = [state[0]/1000 for state in ephemeris_state]
        y_values = [state[1]/1000 for state in ephemeris_state]
        z_values = [state[2]/1000 for state in ephemeris_state]

        # Create 3D figure and plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_values, y_values, z_values)
        omm_utils.plot_earth(ax)
        ax.scatter(initial_state[0]/1000, initial_state[1]/1000, initial_state[2]/1000, label ='True Initial State', s = 10)
        ax.scatter(initial_state_updated[0]/1000, initial_state_updated[1]/1000, initial_state_updated[2]/1000, label ='Estimated Initial State', s = 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'NORAD_ID: {norad_id}')
        ax.legend()
        plt.show()

        # Print Residuals History
        residual_history = estimation_output.residual_history

        # Number of columns and rows for our plot
        number_of_columns = 2

        number_of_rows = (
            int(number_of_pod_iterations / number_of_columns)
            if number_of_pod_iterations % number_of_columns == 0
            else int((number_of_pod_iterations + 1) / number_of_columns)
        )

        fig, axs = plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=(9, 3.5 * number_of_rows),
            sharex=True,
            sharey=False,
        )

        # We cheat a little to get an approximate year out of our times (which are in seconds since J2000)
        residual_times = (
                np.array(norad_id_simulated_observations.concatenated_times) / (86400 * 365.25) + 2000
        )


        # plot the residuals, split between RA and DEC types
        for idx, ax in enumerate(fig.get_axes()):
            ax.grid()
            # we take every second
            ax.scatter(
                residual_times[::2],
                residual_history[
                ::2,
                idx,
                ],
                marker="+",
                s=60,
                label="Right Ascension",
            )
            ax.scatter(
                residual_times[1::2],
                residual_history[
                1::2,
                idx,
                ],
                marker="+",
                s=60,
                label="Declination",
            )
            ax.set_ylabel("Observation Residual [rad]")
            ax.set_title("Iteration " + str(idx + 1))

        plt.tight_layout()

        # add the year label for the x-axis
        for col in range(number_of_columns):
            axs[int(number_of_rows - 1), col].set_xlabel("Year")

        axs[0, 0].legend()

        plt.show()

        # Retrieve and plot correlation matrix
        # Correlation can be retrieved using the CovarianceAnalysisInput class:
        covariance_input = estimation_analysis.CovarianceAnalysisInput(norad_id_simulated_observations)
        covariance_output = estimator.compute_covariance(covariance_input)

        correlations = covariance_output.correlations
        estimated_param_names = ["x", "y", "z", "vx", "vy", "vz"]
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
        ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

        # add numbers to each of the boxes
        for i in range(len(estimated_param_names)):
            for j in range(len(estimated_param_names)):
                text = ax.text(
                    j, i, round(correlations[i, j], 2), ha="center", va="center", color="w"
                )

        cb = plt.colorbar(im)
        ax.set_xlabel("Estimated Parameter")
        ax.set_ylabel("Estimated Parameter")
        fig.suptitle(f"Correlations for estimated parameters for {norad_id}")
        fig.set_tight_layout(True)
        plt.show()

