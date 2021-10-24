"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

TUDATPY EXAMPLE APPLICATION: Earth-Moon Transfer with non-default environment settings
"""
################################################################################
# IMPORT STATEMENTS ############################################################
################################################################################
import numpy as np
from tudatpy.util import result2array
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup, environment
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation

################################################################################
# GENERAL SIMULATION SETUP #####################################################
################################################################################

# Load spice kernels.
spice_interface.load_standard_kernels()

# Set simulation start epoch.
simulation_start_epoch = 1.0e7

# Set numerical integration fixed step size.
fixed_step_size = 3600.0

# Set simulation end epoch.
simulation_end_epoch = 1.0e7 + 5.0 * constants.JULIAN_YEAR

# Set vehicle mass.
vehicle_mass = 2000.0

# Set vehicle thrust magnitude.
thrust_magnitude = 25.0

# Set vehicle specific impulse.
specific_impulse = 5.0e3

################################################################################
# SETUP ENVIRONMENT ############################################################
################################################################################

# create empty body settings dict
bodies_to_create = ["Earth", "Moon", "Sun"]

# create Earth body for simulation
body_settings = environment_setup.get_default_body_settings(bodies_to_create, "SSB", "J2000")

# set constant ephemeris settings for Earth
body_settings.get("Earth").ephemeris_settings = environment_setup.ephemeris.constant(np.zeros(6), "SSB", "J2000")

# set gravity field settings for Earth
body_settings.get("Earth").gravity_field_settings = environment_setup.gravity_field.central_spice()

# create system of bodies in simulation
system_of_bodies = environment_setup.create_system_of_bodies(body_settings)

################################################################################
# SETUP ENVIRONMENT : CREATE VEHICLE ###########################################
################################################################################

system_of_bodies.create_empty_body("Vehicle")
system_of_bodies.get_body("Vehicle").set_constant_mass(vehicle_mass)

################################################################################
# SETUP PROPAGATION : DEFINE THRUST GUIDANCE SETTINGS ##########################
################################################################################

thrust_direction_settings = (
    propagation_setup.thrust.thrust_direction_from_state_guidance(
        central_body="Earth",
        is_colinear_with_velocity=True,
        direction_is_opposite_to_vector=False,
    )
)

thrust_magnitude_settings = (
    propagation_setup.thrust.constant_thrust_magnitude(
        thrust_magnitude=thrust_magnitude, specific_impulse=specific_impulse
    )
)

################################################################################
# SETUP PROPAGATION : CREATE ACCELERATION MODELS ###############################
################################################################################

acceleration_on_vehicle = dict(
    Vehicle=[
        propagation_setup.acceleration.thrust_from_direction_and_magnitude(
            thrust_direction_settings=thrust_direction_settings,
            thrust_magnitude_settings=thrust_magnitude_settings,
        )
    ],
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
)

bodies_to_propagate = ["Vehicle"]

central_bodies = ["Earth"]

acceleration_dict = dict(Vehicle=acceleration_on_vehicle)

# Convert acceleration mappings into acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    body_system=system_of_bodies,
    selected_acceleration_per_body=acceleration_dict,
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies,
)

################################################################################
# SETUP PROPAGATION : PROPAGATION SETTINGS #####################################
################################################################################

# Get gravitational parameter of Earth for initial state.
# gravitational_parameter = system_of_bodies["Earth"].gravity_field_model.get_gravitational_parameter()

# Get system initial state.
system_initial_state = np.array([8.0e6, 0, 0, 0, 7.5e3, 0])
# system_initial_state = elements.keplerian2cartesian(
#     mu=gravitational_parameter,
#     sma=8.0E6,
#     ecc=0.1,
#     inc=np.deg2rad(0.05),
#     raan=np.deg2rad(0.1),
#     argp=np.deg2rad(0.1),
#     theta=np.deg2rad(0.1)
# )

# Create termination settings.
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create propagation settings.
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    termination_condition
)
# Create numerical integrator settings.
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_step_size
)

################################################################################
# PROPAGATE ####################################################################
################################################################################

# Instantiate the dynamics simulator.
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    system_of_bodies, integrator_settings, propagator_settings, True
)

# Propagate and store results to outer loop results dictionary.
result = dynamics_simulator.state_history

################################################################################
# VISUALISATION / OUTPUT / PRELIMINARY ANALYSIS ################################
################################################################################

array = result2array(result)
print(array)
