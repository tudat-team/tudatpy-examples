"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

TUDATPY EXAMPLE APPLICATION: Basic Orbit Estimation
FOCUS:                       orbit estimation of a satellite in MEO
"""

###############################################################################
# TUDATPY EXAMPLE APPLICATION: Keplerian Orbit (two-body problem)  ############
###############################################################################

# TODO: review abstract and inline comments
""" ABSTRACT.

This example implements the following:
- Simulate the orbit of a spacecraft
- Create a tracking station on Earth
- Simulate Doppler data at 1 mm/s every 60 seconds, during periods where the spacecraft is at an elevation angle >15 
  degrees, as viewed from the station.
- Perturb the initial state a 100 m (position components) and 0.1 m/s (velocity components)
- Use the simulated data to estimate the spacecraft initial state, drag coefficient and radiation pressure coefficient

"""

###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion
import matplotlib.pyplot as plt


def main():
    # Load spice kernels.
    spice.load_standard_kernels()

    # Set simulation start and end epochs.
    simulation_start_epoch = 0.0
    simulation_end_epoch = 3.0 * constants.JULIAN_DAY

    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    # Create default body settings for selected celestial bodies
    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as
    # global frame origin and orientation. This environment will only be valid
    # in the indicated time range
    # [simulation_start_epoch --- simulation_end_epoch]
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        "Earth", "J2000")

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    #    print( bodies.get_body('Mars').state_in_base_frame_from_ephemeris( 0.0 ) )

    station_altitude = 0.0
    delft_latitude = np.deg2rad(52.00667)
    delft_longitude = np.deg2rad(4.35556)
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),
        'TrackingStation',
        [station_altitude, delft_latitude, delft_longitude],
        element_conversion.geodetic_position_type)

    ###########################################################################
    # CREATE VEHICLE ##########################################################
    ###########################################################################

    # Create vehicle objects.
    bodies.create_empty_body("Delfi-C3")
    bodies.get_body("Delfi-C3").set_constant_mass(400.0)

    # Create aerodynamic coefficient interface settings, and add to vehicle
    reference_area = 4.0
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0, 0],
        are_coefficients_in_aerodynamic_frame=True,
        are_coefficients_in_negative_axis_direction=True
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "Delfi-C3", aero_coefficient_settings)

    # Create radiation pressure settings, and add to vehicle
    reference_area_radiation = 4.0
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
        bodies, "Delfi-C3", radiation_pressure_settings)

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    # Define bodies that are propagated.
    bodies_to_propagate = ["Delfi-C3"]

    # Define central bodies.
    central_bodies = ["Earth"]

    # Define accelerations acting on Delfi-C3 by Sun and Earth.
    accelerations_settings_delfi_c3 = dict(
        Sun=
        [
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Mars=
        [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Moon=
        [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=
        [
            propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
            propagation_setup.acceleration.aerodynamic()
        ])

    # Create global accelerations settings dictionary.
    acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    ###########################################################################
    # CREATE PROPAGATION SETTINGS #############################################
    ###########################################################################

    # Set initial conditions for the Asterix satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Keplerian elements and later on converted to Cartesian elements.
    earth_gravitational_parameter = bodies.get_body("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=7500.0E3,
        eccentricity=0.1,
        inclination=np.deg2rad(85.3),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87)
    )

    # Create propagation settings.
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_condition
    )
    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch, 60.0, propagation_setup.integrator.rkf_78, 60.0, 60.0, 1.0, 1.0
    )

    # Create list of parameters that are to be estimated
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies)
    parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi-C3"))
    parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("Delfi-C3"))

    parameter_set = estimation_setup.create_parameters_to_estimate(parameter_settings, bodies)

    ###########################################################################
    # DEFINE LINKS ############################################################
    ###########################################################################

    # Define uplink link ends for one-way observable
    link_ends = dict()
    link_ends[observation.transmitter] = ('Earth', 'TrackingStation')
    link_ends[observation.receiver] = ('Delfi-C3', '')

    # Create observation settings for each link/observable
    observation_settings_list = [observation.one_way_open_loop_doppler(link_ends)]

    ###########################################################################
    # CREATE ESTIMATION OBJECT ################################################
    ###########################################################################

    estimator = numerical_simulation.Estimator(
        bodies, parameter_set, observation_settings_list, integrator_settings,
        propagator_settings)

    ###########################################################################
    # SIMULATE OBSERVATIONS ###################################################
    ###########################################################################

    # Define observation simulation times for each link
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 60.0)
    observation_simulation_settings = observation.tabulated_simulation_settings(
        observation.one_way_doppler_type,
        link_ends,
        observation_times
    )

    # Add noise levels
    noise_level = 1.0E-3 / constants.SPEED_OF_LIGHT
    observation.add_gaussian_noise_to_settings(
        [observation_simulation_settings],
        noise_level,
        observation.one_way_doppler_type
    )

    # Create viability settings
    viability_setting = observation.elevation_angle_viability(["Earth", "TrackingStation"], np.deg2rad(15))
    observation.add_viability_check_to_settings(
        [observation_simulation_settings],
        [viability_setting]
    )

    # Simulate required observation
    simulated_observations = observation.simulate_observations(
        [observation_simulation_settings],
        estimator.observation_simulators,
        bodies)

    # Perturb initial positio  by 1 m in each direction
    initial_parameter_deviation = np.zeros(parameter_set.parameter_set_size)
    initial_parameter_deviation[0:3] = 100.0
    initial_parameter_deviation[3:6] = 0.1

    truth_parameters = parameter_set.parameter_vector

    # Create input for estimation
    pod_input = estimation.PodInput(
        simulated_observations, parameter_set.parameter_set_size,
        apriori_parameter_correction=initial_parameter_deviation)
    pod_input.define_estimation_settings(
        reintegrate_variational_equations=False)

    # Define weights
    weights_per_observable = \
        {estimation_setup.observation.one_way_doppler_type: noise_level ** -2}
    pod_input.set_constant_weight_per_observable(weights_per_observable)

    # Perform estimation
    pod_output = estimator.perform_estimation(pod_input)
    print(pod_output.formal_errors)
    print(truth_parameters - parameter_set.parameter_vector)

    # Save some figures to files as test
    plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()

    observation_times = np.array(simulated_observations.concatenated_times)
    observations_list = np.array(simulated_observations.concatenated_observations)

    plt.figure(figsize=(17, 5))
    plt.title("Observations as a function of time.")
    plt.scatter(observation_times / 3600.0, observations_list * constants.SPEED_OF_LIGHT)
    plt.xlabel('Time [hr]')
    plt.ylabel('Range rate [m/s]')
    plt.grid()
    plt.show()

    residual_history = pod_output.residual_history
    subplots_list = list()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 17))
    subplots_list.append(ax1)
    subplots_list.append(ax2)
    subplots_list.append(ax3)
    subplots_list.append(ax4)
    subplots_list.append(ax5)

    for i in range(5):
        subplots_list[i].scatter(observation_times, residual_history[:, i])
    plt.show()

    final_residuals = pod_output.final_residuals
    plt.figure(figsize=(17, 5))
    plt.hist(final_residuals, 25)
    plt.show()

    print(pod_output.formal_errors / (truth_parameters - parameter_set.parameter_vector))

    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()
