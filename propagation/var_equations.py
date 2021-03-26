
# # # # # # # # # # # #                           # # # # # # # # # # # #
#               - variational equations example usecase -               #
#    Propagation of a variational equations alongside the perturbed     #
#           Earth orbiter from the minimal example usecase.             #
# # # # # # # # # # # #                           # # # # # # # # # # # #
#                                                                       #
# + This script is an simple extension of the minimal example usecase,  #
#   including the propagation of variational equations alongside the    #
#   perturbed Earth orbiter.                                            #
#                                                                       #
# + This simple extension already allows for sensitivity study of the   #
#   trajectory w.r.t the orbiter's initial state, vehicle properties    #
#   (e.g. drag coefficients, reference areas) and even environmental    #
#   parameters such as gravitational, atmospheric or irradiation        #
#   parameters of the celestial bodies in the system.                   #
#                                                                       #
# + An extensive guide to this example application can be found in the  #
#   docs under "Getting Started with Simulations".                      #


"""
!! Large portions of the problem setup are identical to the minimal
    usecase of the perturbed earth orbiter.
!! The extensions for the propagation of variational equations are
    highlighted in triple double-quoted string comments.
"""

###########################################################################
# # # # # # # # # # # #      IMPORT STATEMENTS      # # # # # # # # # # # #
###########################################################################

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.simulation import estimation_setup
from tudatpy.kernel.astro import conversion


def main():

    #######################################################################
    # # # # # # # # # # #         SPICE SETUP         # # # # # # # # # # #
    #######################################################################

    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start and end epochs.
    simulation_start_epoch = 0.0
    simulation_end_epoch = constants.JULIAN_DAY

    #######################################################################
    # # # # # # # # # # #      ENVIRONMENT SETUP      # # # # # # # # # # #
    #######################################################################


    # #########        CREATE BODIES        ######### #

    # Make list of selected celestial bodies to be created.
    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

    # Set global frame orientation (around Sun by default).
    global_frame_orientation = "J2000"

    # Create body settings at default for all bodies_to_create.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        base_frame_orientation=global_frame_orientation
    )

    # Create system of selected celestial bodies from body_settings.
    bodies = environment_setup.create_system_of_bodies(body_settings)


    # #########        CREATE VEHICLE        ######### #

    # Create vehicle object.
    bodies.create_empty_body( "Delfi-C3" )
    bodies.get_body( "Delfi-C3").set_constant_mass( 2.2 )


    # #########      CREATE INTERFACES       ######### #

    # Define aerodynamic interface for vehicle.
    reference_area = 0.05
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area,[drag_coefficient,0,0]
    )
    # Create aerodynamic interface for vehicle.
    environment_setup.add_aerodynamic_coefficient_interface(
                bodies, "Delfi-C3", aero_coefficient_settings
    )

    # Define radiation pressure interface for vehicle.
    reference_area_radiation = 0.05
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    # Create radiation pressure interface for vehicle.
    environment_setup.add_radiation_pressure_interface(
                bodies, "Delfi-C3", radiation_pressure_settings
    )


    #######################################################################
    # # # # # # # # # # #      PROPAGATION SETUP      # # # # # # # # # # #
    #######################################################################

    # Define bodies that are to be propagated.
    bodies_to_propagate = ["Delfi-C3"]
    # Define central bodies (w.r.t. which the vehicle state vector is defined)
    central_bodies = ["Earth"]


    # #########     CREATE ACCELERATION MODELS     ######### #

    # Define accelerations acting on Delfi-C3 by Sun and Earth in a dictionary.
    accelerations_settings_delfi_c3 = dict(
        Sun=
        [
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=
        [
            propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
            propagation_setup.acceleration.aerodynamic()
        ]
    )

    # Define point mass accelerations acting on Delfi-C3 by all other bodies.
    for other in set(bodies_to_create).difference({"Sun", "Earth"}):
        accelerations_settings_delfi_c3[other] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]

    # Create global accelerations settings dict
    #  and register acceleration dict of vehicle.
    acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies
    )


    # #########    DEFINE INITIAL SYSTEM STATE     ######### #

    # Set initial conditions for the bodies to be propagated (vehicle)
    # The initial conditions are given in Keplerian elements
    #  and later on converted to Cartesian elements.
    # For conversion, the gravitational_parameter of the central
    #  body can be retrieved from the the bodies variable.

    earth_gravitational_parameter = bodies.get_body( "Earth" ).gravitational_parameter

    initial_state = conversion.keplerian_to_cartesian(
        gravitational_parameter = earth_gravitational_parameter,
        semi_major_axis = 7500.0E3,
        eccentricity = 0.1,
        inclination = np.deg2rad(85.3),
        argument_of_periapsis = np.deg2rad(235.7),
        longitude_of_ascending_node = np.deg2rad(23.4),
        true_anomaly = np.deg2rad(139.87)
    )


    # #########   SET DEPENDENT VARIABLES TO SAVE  ######### #

    # Define list of dependent variables to save.
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.total_acceleration( "Delfi-C3" ),
        propagation_setup.dependent_variable.keplerian_state( "Delfi-C3", "Earth" ),
        propagation_setup.dependent_variable.latitude( "Delfi-C3", "Earth" ),
        propagation_setup.dependent_variable.longitude( "Delfi-C3", "Earth" ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Sun"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Moon"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Mars"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Venus"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type, "Delfi-C3", "Earth"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, "Delfi-C3", "Earth"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.cannonball_radiation_pressure_type, "Delfi-C3", "Sun"
        )
    ]


    # #########         PROPAGATOR SETTINGS        ######### #

    # Create propagator settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_end_epoch,
        output_variables=dependent_variables_to_save
    )


    # #########         INTEGRATOR SETTINGS        ######### #

    # Define integrator settings.
    fixed_step_size = 10.0
    # Create integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch,
        fixed_step_size
    )



    """
    !! The problem setup is complete and so from here on the script deviates 
        significantly from the minimal example usecase. 
    !! Instead of a dynamics simulator, a variational equations solver object
        is used for the propagation of the orbiter state and the variational 
        equations w.r.t the user-defined variational parameters.
    !! After retrieving the results, the state transition matrix history is 
        used to propagate uncertainties of selected parameters and to assess 
        their impact on the orbiter's trajectory.
    """


    #######################################################################
    # # # # # # # # # # #       SIMULATOR USAGE       # # # # # # # # # # #
    #######################################################################


    # #########        VARIATIONAL PARAMETERS       ######### #

    # Create list of parameters for which the variational equations are to be propagated
    parameter_settings = estimation_setup.parameter.initial_states( propagator_settings, bodies )
    # Add parameters to that list
    parameter_settings.append( estimation_setup.parameter.gravitational_parameter( "Earth" ) )
    parameter_settings.append( estimation_setup.parameter.constant_drag_coefficient( "Delfi-C3" ) )


    # #########  CREATE VARIATIONAL EQUATIONS SOLVER  ######### #

    variational_equations_solver = estimation_setup.SingleArcVariationalEquationsSolver(
        bodies, integrator_settings, propagator_settings, estimation_setup.create_parameters_to_estimate(
            parameter_settings, bodies
        ),
        integrate_on_creation=1
    )

    # #########            RETRIEVE RESULTS           ######### #

    states = variational_equations_solver.state_history
    state_transition_matrices = variational_equations_solver.state_transition_matrix_history
    sensitivity_matrices = variational_equations_solver.sensitivity_matrix_history


    #######################################################################
    # # # # # # # # #      PROPAGATING VARIATIONS     # # # # # # # # # # #
    #######################################################################

    # #########           DEFINE VARIATIONS           ######### #

    # Variation of vehicle initial state
    initial_state_variation = [1, 0, 0, 1.0E-3, 0, 0]
    # Variation of Earth gravitational parameter
    earth_standard_param_variation = [-2.0E+5, 0.0]
    # Variation of vehicle drag coefficient
    drag_coeff_variation = [0.0, 0.05]


    # ######### COMPUTE IMPACT ON ORBITER TRAJECTORY  ######### #

    # make dictionaries to collect results from matrix multiplication
    delta_initial_state_dict = dict()
    earth_standard_param_dict = dict()
    delta_drag_coeff_dict = dict()

    # multiply matrices with variation vectors over all simulation epochs
    for epoch in state_transition_matrices:
        delta_initial_state_dict[epoch] = np.dot(state_transition_matrices[epoch], initial_state_variation)
        earth_standard_param_dict[epoch] = np.dot(sensitivity_matrices[epoch], earth_standard_param_variation)
        delta_drag_coeff_dict[epoch] = np.dot(sensitivity_matrices[epoch], drag_coeff_variation)


    #######################################################################
    # # # # # # # # #         VISUALISE IMPACT        # # # # # # # # # # #
    #######################################################################


    # #########           PYPLOT SETUP           ######### #

    from matplotlib import pyplot as plt
    font_size = 20
    plt.rcParams.update({'font.size': font_size})


    # #########           PREPROCESSING          ######### #

    # Retrieve and convert propagation epochs
    time = state_transition_matrices.keys()
    time_hours = [t / 3600 for t in time]

    # Reformat change of states
    delta_initial_state = np.vstack(list(delta_initial_state_dict.values()))
    delta_earth_standard_param = np.vstack(list(earth_standard_param_dict.values()))
    delta_drag_coefficient = np.vstack(list(delta_drag_coeff_dict.values()))

    # Compute deviations of position and speed
    # 1 // due to initial state variation
    delta_r1 = np.sqrt(delta_initial_state[:, 0] ** 2 + delta_initial_state[:, 1] ** 2 + delta_initial_state[:, 2] ** 2)
    delta_v1 = np.sqrt(delta_initial_state[:, 3] ** 2 + delta_initial_state[:, 4] ** 2 + delta_initial_state[:, 5] ** 2)
    # 2 // due to gravitational parameter variation
    delta_r2 = np.sqrt(delta_earth_standard_param[:, 0] ** 2 + delta_earth_standard_param[:, 1] ** 2 + delta_earth_standard_param[:, 2] ** 2)
    delta_v2 = np.sqrt(delta_earth_standard_param[:, 3] ** 2 + delta_earth_standard_param[:, 4] ** 2 + delta_earth_standard_param[:, 5] ** 2)
    # 3 // due to drag coefficient variation
    delta_r3 = np.sqrt(delta_drag_coefficient[:, 0] ** 2 + delta_drag_coefficient[:, 1] ** 2 + delta_drag_coefficient[:, 2] ** 2)
    delta_v3 = np.sqrt(delta_drag_coefficient[:, 3] ** 2 + delta_drag_coefficient[:, 4] ** 2 + delta_drag_coefficient[:, 5] ** 2)


    # #########           CREATE FIGURES         ######### #

    # Plot deviations of position
    plt.figure( figsize=(17,5))
    plt.grid()
    plt.plot(time_hours, delta_r1, color='tomato', label='variation initial state')
    plt.plot(time_hours, delta_r2, color='orange', label='variation grav. parameter (Earth)')
    plt.plot(time_hours, delta_r3, color='cyan', label='variation drag coefficient')
    plt.yscale('log')
    plt.xlabel('Time [hr]')
    plt.ylabel('$\Delta r (t_1)$ [m]')
    plt.xlim( [min(time_hours), max(time_hours)] )
    plt.legend()
    plt.savefig(fname='position_deviation.png', bbox_inches='tight')

    # Plot deviations of speed
    plt.figure( figsize=(17,5))
    plt.grid()
    plt.plot(time_hours, delta_v1, color='tomato', label='variation initial state')
    plt.plot(time_hours, delta_v2, color='orange', label='variation grav. parameter (Earth)')
    plt.plot(time_hours, delta_v3, color='cyan', label='variation drag coefficient')
    plt.yscale('log')
    plt.xlabel('Time [hr]')
    plt.ylabel('$\Delta v (t_1)$ [m/s]')
    plt.xlim( [min(time_hours), max(time_hours)] )
    plt.legend()
    plt.savefig(fname='velocity_deviation.png', bbox_inches='tight')


    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()


###########################################################################
# # # # # # # # # # # #           THE END           # # # # # # # # # # # #
###########################################################################