###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################
import sys
sys.path.insert(0, '/home/dominic/Software/tudat-bundle/build-tudat-bundle-Desktop-Default/tudatpy/')

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observations
from tudatpy.kernel.astro import element_conversion
import matplotlib.pyplot as plt

def main():
    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start and end epochs.
    simulation_start_epoch = 0.0
    simulation_end_epoch = constants.JULIAN_DAY

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
        "Earth","J2000")

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
#    print( bodies.get_body('Mars').state_in_base_frame_from_ephemeris( 0.0 ) )
    print( bodies.get_body('Moon').gravitational_parameter )

    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station1',[0.0,1.25,0.0],element_conversion.geodetic_position_type)
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station2',[0.0,-1.55,2.0],element_conversion.geodetic_position_type)
    environment_setup.add_ground_station(
        bodies.get_body('Earth'),'Station3',[0.0,0.8,4.0],element_conversion.geodetic_position_type)

    ###########################################################################
    # CREATE VEHICLE ##########################################################
    ###########################################################################

    # Create vehicle objects.
    bodies.create_empty_body( "Delfi-C3" )
    bodies.get_body( "Delfi-C3").set_constant_mass(400.0)

    # Create aerodynamic coefficient interface settings, and add to vehicle
    reference_area = 4.0
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,[drag_coefficient,0,0],
        are_coefficients_in_aerodynamic_frame=True,
        are_coefficients_in_negative_axis_direction=True
    )
    environment_setup.add_aerodynamic_coefficient_interface(
                bodies, "Delfi-C3", aero_coefficient_settings );

    # Create radiation pressure settings, and add to vehicle
    reference_area_radiation = 4.0
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
                bodies, "Delfi-C3", radiation_pressure_settings )
    environment_setup.add_empty_tabulated_ephemeris( bodies, "Delfi-C3" )

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
    earth_gravitational_parameter = bodies.get_body( "Earth" ).gravitational_parameter
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
        simulation_start_epoch, 40.0, propagation_setup.integrator.rkf_78, 40.0, 40.0, 1.0, 1.0
    )

    # Create list of parameters for which the variational equations are to be
    # propagated
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies)
    parameter_settings.append(
        estimation_setup.parameter.gravitational_parameter("Earth"))
    parameter_settings.append(
        estimation_setup.parameter.constant_drag_coefficient("Delfi-C3"))
    parameter_settings.append(
        estimation_setup.parameter.radiation_pressure_coefficient("Delfi-C3"))

    parameter_set = estimation_setup.create_parameters_to_estimate( parameter_settings, bodies )

    ###########################################################################
    # DEFINE LINKS ############################################################
    ###########################################################################

    receiver_link_ends = list()
    transmitter_link_ends = list()

    # Define up- and downlink link ends for one-way observable
    for i in range(3):
        linkEnds = dict()
        linkEnds[observations.transmitter ] = ( 'Earth', 'Station' + str(i+1))
        linkEnds[observations.receiver ] = ( 'Delfi-C3', '' )
        transmitter_link_ends.append( linkEnds )

        linkEnds.clear( )
        linkEnds[observations.receiver ] = ( 'Earth', 'Station' + str(i+1))
        linkEnds[observations.transmitter ] = ( 'Delfi-C3', '' )
        receiver_link_ends.append( linkEnds )

    # Set (semi-random) combination of link ends for obsevables
    link_ends_per_observable = dict()
    link_ends_per_observable[ observations.one_way_range_type ] = list()
    link_ends_per_observable[ observations.one_way_range_type ].append( receiver_link_ends[ 0 ] )
    link_ends_per_observable[ observations.one_way_range_type ].append( transmitter_link_ends[ 0 ] )
    link_ends_per_observable[ observations.one_way_range_type ].append( receiver_link_ends[ 1 ] )

    # Create observation settings for each link/observable
    observation_settings_list = list()

    # Iterate over all observables
    for observable_key in link_ends_per_observable:
        current_link_ends = link_ends_per_observable[ observable_key ]
        # Iterate over all link ends
        for i in range( len(current_link_ends) ):
            # Create observation settings
            if( observable_key ==observations.one_way_range_type ):
                observation_settings_list.append( observations.one_way_range( current_link_ends[i] ) )

    ###########################################################################
    # CREATE ESTIMATION OBJECT ################################################
    ###########################################################################

    orbit_determination_manager = numerical_simulation.Estimator(
        bodies,parameter_set,observation_settings_list,integrator_settings,propagator_settings)

    ###########################################################################
    # SIMULATE OBSERVATIONS ###################################################
    ###########################################################################

    # Simulate observations
    single_arc_observation_times = np.arange( 0.0, 10000, 20 )
    total_observation_times = single_arc_observation_times
    # total_observation_times = np.concatenate(
    #     [ single_arc_observation_times, single_arc_observation_times + 86400.0, single_arc_observation_times + 2.0 * 86400.0 ] )

    # Define observation simulation times for each link
    observation_simulation_settings = observations.create_tabulated_simulation_settings( link_ends_per_observable,total_observation_times )

    # Simulate required observation
    simulated_observations = observations.simulate_observations(
        observation_simulation_settings,
        orbit_determination_manager.observation_simulators,
        bodies )

    # Perturb initial positio  by 1 m in each direction
    initial_parameter_deviation = np.zeros( 9 )
    initial_parameter_deviation[ 0 ] = 1.0
    initial_parameter_deviation[ 1 ] = 1.0
    initial_parameter_deviation[ 2 ] = 1.0

    # Create unput for estimation
    pod_input = estimation.PodInput(
        simulated_observations, parameter_set.parameter_set_size,
        apriori_parameter_correction = initial_parameter_deviation  )
    pod_input.define_estimation_settings(
        reintegrate_variational_equations = False )

    # Define weigh
    weights_per_observable = dict()
    weights_per_observable[ estimation_setup.observations.one_way_range_type ] = 1.0 / ( 0.1*0.1 )
    pod_input.set_constant_weight_per_observable(
        weights_per_observable)

    # Perform estimation
    pod_output = orbit_determination_manager.perform_estimation( pod_input )
    print(pod_output.formal_errors)

    # Save some figures to files as test
    plt.imshow(pod_output.correlations, aspect='auto', interpolation='none')
    plt.show()

    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()
