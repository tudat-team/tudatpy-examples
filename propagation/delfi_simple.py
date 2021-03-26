
# # # # # # # # # # # #                           # # # # # # # # # # # #
#                  - minimal tudatpy example usecase -                  #
#    Propagation of a spacecraft in Earth orbit under perturbations     #
#    from third bodies, solar radiation pressure and the atmosphere.    #
# # # # # # # # # # # #                           # # # # # # # # # # # #
#                                                                       #
# + This script is provided to support a hands-on introduction to       #
#   tudatpy and can serve as a foundation for the development of        #
#   more advanced use cases.                                            #
#                                                                       #
# + An extensive guide to this example application can be found in the  #
#   docs under "Getting Started with Simulations".                      #


###########################################################################
# # # # # # # # # # # #      IMPORT STATEMENTS      # # # # # # # # # # # #
###########################################################################

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
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


    #######################################################################
    # # # # # # # # # # #       SIMULATOR USAGE       # # # # # # # # # # #
    #######################################################################


    # #########     CREATE DYNAMICS SIMULATOR       ######### #

    # Create simulation object and propagate dynamics.
    dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
        bodies, integrator_settings, propagator_settings)


    # #########           RETRIEVE RESULTS          ######### #

    states = dynamics_simulator.state_history
    dependent_variables = dynamics_simulator.dependent_variable_history


    # #########   PRINT INITIAL AND FINAL STATES    ######### #

    print(
        f"""
    Delfi-C3.
    The initial position vector of Delfi-C3 is [km]: \n{
        states[simulation_start_epoch][:3] / 1E3}
    The initial velocity vector of Delfi-C3 is [km/s]: \n{
        states[simulation_start_epoch][3:] / 1E3}
    After {simulation_end_epoch} seconds the position vector of Delfi-C3 is [km]: \n{
        states[simulation_end_epoch][:3] / 1E3}
    And the velocity vector of Delfi-C3 is [km/s]: \n{
        states[simulation_end_epoch][3:] / 1E3}
        """
    )


    #######################################################################
    # # # # # # # # # # #      VISUALISE RESULTS      # # # # # # # # # # #
    #######################################################################


    # #########           PYPLOT SETUP           ######### #

    from matplotlib import pyplot as plt
    font_size = 20
    plt.rcParams.update({'font.size': font_size})


    # #########           PREPROCESSING          ######### #

    # Retrieve and convert propagation epochs (time)
    time = dependent_variables.keys()
    time_hours = [t / 3600 for t in time]

    # Collect dependent variables in numpy array:
    dependent_variable_list = np.vstack(list(dependent_variables.values()))
    #       0-2: total acceleration
    #       3-8: Keplerian state
    #       9: latitude
    #       10: longitude
    #       11: Acceleration Norm PM Sun
    #       12: Acceleration Norm PM Moon
    #       13: Acceleration Norm PM Mars
    #       14: Acceleration Norm PM Venus
    #       15: Acceleration Norm SH Earth


    # #########          CREATE FIGURES          ######### #

    # Total Acceleration
    total_acceleration = np.sqrt(
        dependent_variable_list[:, 0] ** 2 + dependent_variable_list[:, 1] ** 2 + dependent_variable_list[:, 2] ** 2)
    # - #
    plt.figure(figsize=(17, 5))
    plt.grid()
    plt.plot(time_hours, total_acceleration)
    plt.xlabel('Time [hr]')
    plt.ylabel('Total Acceleration [m/s$^2$]')
    plt.xlim([min(time_hours), max(time_hours)])
    plt.savefig(fname='total_acceleration.png', bbox_inches='tight')

    # Ground Track
    latitude = dependent_variable_list[:, 9]
    longitude = dependent_variable_list[:, 10]
    part = int(len(time) / 24 * 3)
    latitude = np.rad2deg(latitude[0:part])
    longitude = np.rad2deg(longitude[0:part])
    # - #
    plt.figure(figsize=(17, 5))
    plt.grid()
    plt.yticks(np.arange(-90, 91, step=45))
    plt.scatter(longitude, latitude, s=1)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.xlim([min(longitude), max(longitude)])
    plt.savefig(fname='ground_track.png', bbox_inches='tight')

    # Kepler Elements
    kepler_elements = dependent_variable_list[:, 3:9]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 17))
    #       Semi-major Axis
    semi_major_axis = [element / 1000 for element in kepler_elements[:, 0]]
    ax1.plot(time_hours, semi_major_axis)
    ax1.set_ylabel('Semi-major axis [km]')
    #       Eccentricity
    eccentricity = kepler_elements[:, 1]
    ax2.plot(time_hours, eccentricity)
    ax2.set_ylabel('Eccentricity [-]')
    #       Inclination
    inclination = [np.rad2deg(element) for element in kepler_elements[:, 2]]
    ax3.plot(time_hours, inclination)
    ax3.set_ylabel('Inclination [deg]')
    #       Argument of Periapsis
    argument_of_periapsis = [np.rad2deg(element) for element in kepler_elements[:, 3]]
    ax4.plot(time_hours, argument_of_periapsis)
    ax4.set_ylabel('Argument of Periapsis [deg]')
    #       Right Ascension of the Ascending Node
    raan = [np.rad2deg(element) for element in kepler_elements[:, 4]]
    ax5.plot(time_hours, raan)
    ax5.set_ylabel('RAAN [deg]')
    #       True Anomaly
    true_anomaly = [np.rad2deg(element) for element in kepler_elements[:, 5]]
    ax6.scatter(time_hours, true_anomaly, s=1)
    ax6.set_ylabel('True Anomaly [deg]')
    ax6.set_yticks(np.arange(0, 361, step=60))
    # - #
    for ax in fig.get_axes():
        ax.set_xlabel('Time [hr]')
        ax.set_xlim([min(time_hours), max(time_hours)])
        ax.grid()
    # - #
    fig.savefig(fname='kepler_elements.png', bbox_inches='tight')


    # Accelerations by body and type
    plt.figure(figsize=(17, 5))
    #       Point Mass Gravity Acceleration Sun
    acceleration_norm_pm_sun = dependent_variable_list[:, 11]
    plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')
    #       Point Mass Gravity Acceleration Moon
    acceleration_norm_pm_moon = dependent_variable_list[:, 12]
    plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')
    #       Point Mass Gravity Acceleration Mars
    acceleration_norm_pm_mars = dependent_variable_list[:, 13]
    plt.plot(time_hours, acceleration_norm_pm_mars, label='PM Mars')
    #       Point Mass Gravity Acceleration Venus
    acceleration_norm_pm_venus = dependent_variable_list[:, 14]
    plt.plot(time_hours, acceleration_norm_pm_venus, label='PM Venus')
    #       Spherical Harmonic Gravity Acceleration Earth
    acceleration_norm_sh_earth = dependent_variable_list[:, 15]
    plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')
    #       Aerodynamic Acceleration Earth
    acceleration_norm_aero_earth = dependent_variable_list[:, 16]
    plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')
    #       Cannonball Radiation Pressure Acceleration Sun
    acceleration_norm_rp_sun = dependent_variable_list[:, 17]
    plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')
    # - #
    plt.grid()
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.xlim([min(time_hours), max(time_hours)])
    plt.yscale('log')
    plt.xlabel('Time [hr]')
    plt.ylabel('Acceleration Norm [m/s$^2$]')
    # - #
    plt.savefig(fname='acceleration_norms.png', bbox_inches='tight')


    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()


###########################################################################
# # # # # # # # # # # #           THE END           # # # # # # # # # # # #
###########################################################################
