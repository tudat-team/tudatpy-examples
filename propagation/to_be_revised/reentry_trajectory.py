"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

TUDATPY EXAMPLE APPLICATION: Re-entry trajectory
"""

###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################

import sys
sys.path.insert(0, '/home/dominic/Software/tudat-bundle/build-tudat-bundle-Desktop-Default/tudatpy/')

import numpy as np
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel.astro import element_conversion
from matplotlib import pyplot as plt


class CapsuleAerodynamicGuidance(propagation.AerodynamicGuidance):

    def __init__(self,
                 bodies: environment.SystemOfBodies):
        # Call the base class constructor
        propagation.AerodynamicGuidance.__init__(self)

        self.vehicle = bodies.get_body('STS')
        self.earth = bodies.get_body('Earth')

        self.vehicle_flight_conditions = bodies.get_body('STS').flight_conditions
        self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
        self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface


    def updateGuidance(self,
                       current_time: float):

        earth_angular_velocity = np.linalg.norm(self.earth.body_fixed_angular_velocity)
        body_position = self.vehicle.position
        earth_distance = np.linalg.norm(body_position)
        body_mass = self.vehicle.mass

        mach_number = self.vehicle_flight_conditions.mach_number
        airspeed = self.vehicle_flight_conditions.airspeed
        density = self.vehicle_flight_conditions.density

        # set AOA
        if mach_number > 12:
            self.angle_of_attack = np.deg2rad(40.)
        elif mach_number < 6:
            self.angle_of_attack = np.deg2rad(10.)
        else:
            self.angle_of_attack = np.deg2rad(40. - 5. * (12 - mach_number))

        current_aerodynamics_independent_variables = [self.angle_of_attack,mach_number]
        self.aerodynamic_coefficient_interface.update_coefficients(
            current_aerodynamics_independent_variables, current_time)

        current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
        aerodynamic_reference_area = self.aerodynamic_coefficient_interface.reference_area

        heading = self.aerodynamic_angle_calculator.get_angle(environment.heading_angle)
        flight_path_angle = self.aerodynamic_angle_calculator.get_angle(environment.flight_path_angle)
        latitude = self.aerodynamic_angle_calculator.get_angle(environment.latitude_angle)

        lift_acceleration = 0.5 * density * airspeed * airspeed * aerodynamic_reference_area * current_force_coefficients[2] / body_mass
        downward_gravitational_acceleration = self.earth.gravitational_parameter / (earth_distance * earth_distance)
        spacecraft_centrifugal_acceleration = airspeed * airspeed / earth_distance
        coriolis_acceleration = 2.0 * earth_angular_velocity * airspeed * np.cos( latitude ) * np.sin( heading )
        earth_centrifugal_acceleration = earth_angular_velocity * earth_angular_velocity * earth_distance * np.cos( latitude) * \
                                   (np.cos(latitude) * np.cos(flight_path_angle) + np.sin(flight_path_angle) * np.sin(latitude) * np.cos( heading))
        
        cosine_of_bank_angle = ( (downward_gravitational_acceleration - spacecraft_centrifugal_acceleration ) * np.cos( flight_path_angle) - coriolis_acceleration - earth_centrifugal_acceleration ) / lift_acceleration
        if (cosine_of_bank_angle < -1.0):
            self.bank_angle = np.deg2rad(180.0)
        elif (cosine_of_bank_angle > 1.0):
            self.bank_angle = 0.0
        else:
            self.bank_angle = np.arccos(cosine_of_bank_angle)

        self.sideslip_angle = 0.0

def main():



    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start epoch.
    simulation_start_epoch = 0.0

    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    # Create default body settings for "Earth"
    bodies_to_create = ["Earth"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as
    # global frame origin and orientation
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, "Earth", "J2000"
    )
    # body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.constant('J2000','IAU_Earth',np.identity(3))

    # Create Earth Object.
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ###########################################################################
    # CREATE VEHICLE ##########################################################
    ###########################################################################

    # Create vehicle object.
    bodies.create_empty_body("STS")
    bodies.get_body( "STS" ).set_constant_mass(5.0e3)

    # Create vehicle object.
    force_coefficient_files = dict()
    force_coefficient_files[ 0 ] = 'input/STS_CD.dat'
    force_coefficient_files[ 2 ] = 'input/STS_CL.dat'


    # Add predefined aerodynamic coefficient database to the body
    coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
        force_coefficient_files, 2690.0 * 0.3048 * 0.3048, [ environment.angle_of_attack_dependent, environment.mach_number_dependent ], True, True )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "STS", coefficient_settings)

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    # Define bodies that are propagated.
    bodies_to_propagate = ["STS"]

    # Define central bodies.
    central_bodies = ["Earth"]

    # Define accelerations acting on STS.
    accelerations_settings_STS = dict(
        Earth=[
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.aerodynamic(),
        ]
    )
    acceleration_settings = {"STS": accelerations_settings_STS}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    ###########################################################################
    # CREATE PROPAGATION SETTINGS #############################################
    ###########################################################################

    # Create new aerodynamic guidance
    guidance_object = CapsuleAerodynamicGuidance(bodies)
    # Set aerodynamic guidance (this line links the CapsuleAerodynamicGuidance settings with the propagation)
    environment_setup.set_aerodynamic_guidance(guidance_object,
                                               bodies.get_body('STS'))

    # Set spherical elements for STS and convert to Cartesian.
    initial_radial_distance = (
        bodies.get_body("Earth").shape_model.average_radius + 120.0e3
    )
    initial_earth_fixed_state = element_conversion.spherical_to_cartesian_elementwise(
        radial_distance=initial_radial_distance,
        latitude=0.3,
        longitude=1.2,
        speed=7.5e3,
        flight_path_angle=np.deg2rad(-0.6),
        heading_angle=0.6,
    )

    # Convert the state from Earth-fixed to inertial frame
    earth_rotation_model = bodies.get_body("Earth").rotation_model
    initial_state = environment.transform_to_inertial_orientation(
        initial_earth_fixed_state, simulation_start_epoch, earth_rotation_model
    )

    # Define list of dependent variables to save.
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.flight_path_angle("STS", "Earth"),
        propagation_setup.dependent_variable.altitude("STS", "Earth"),
        propagation_setup.dependent_variable.bank_angle("STS", "Earth"),
        propagation_setup.dependent_variable.angle_of_attack("STS", "Earth"),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, "STS", "Earth"
        ),
        propagation_setup.dependent_variable.aerodynamic_force_coefficients("STS"),
    ]

    # Define termination conditions (once altitude goes below 25 km).
    termination_variable = propagation_setup.dependent_variable.altitude(
        "STS", "Earth"
    )
    termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=termination_variable,
        limit_value=25.0e3,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False,
    )

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )

    # Create numerical integrator settings.
    fixed_step_size = 0.1
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        simulation_start_epoch, fixed_step_size
    )

    ###########################################################################
    # PROPAGATE ORBIT #########################################################
    ###########################################################################

    # Create simulation object and propagate dynamics.
    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, integrator_settings, propagator_settings
    )
    states = dynamics_simulator.state_history
    dependent_variables = dynamics_simulator.dependent_variable_history

    time = dependent_variables.keys()
    time_hours = [t / 3600 for t in time]
    dependent_variables_list = np.vstack(list(dependent_variables.values()))



    flight_path_angle = dependent_variables_list[:, 0]
    altitude = dependent_variables_list[:, 1]
    bank_angle = dependent_variables_list[:, 2]
    angle_of_attack = dependent_variables_list[:, 3]
    aerodynamic_acceleration = dependent_variables_list[:, 4]
    drag_coefficient = dependent_variables_list[:, 5]
    lift_coefficient = dependent_variables_list[:, 7]

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, np.rad2deg(flight_path_angle))
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Flight-path angle [deg]')

    flight_path_angle_derivative = np.absolute(( flight_path_angle[1:flight_path_angle.size] - flight_path_angle[0:-1])/fixed_step_size)
    plt.figure(figsize=(17, 5))
    plt.plot(time_hours[0:-1], np.rad2deg(flight_path_angle_derivative))
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Absolute flight-path angle rate [deg/s]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, altitude)
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Altitude [m]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, np.rad2deg(bank_angle))
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Bank angle [deg]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, np.rad2deg(angle_of_attack))
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Angle of attack [deg]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, aerodynamic_acceleration)
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Aerodynamic acceleration [m/s^2]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, drag_coefficient)
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Lift coefficient [-]')

    plt.figure(figsize=(17, 5))
    plt.plot(time_hours, lift_coefficient)
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('Drag coefficient [-]')

    plt.show()

    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()

