# Earth-Mars transfer window design using Porkchop Plots
"""

Copyright (c) 2010-2024, Delft University of Technology
All rigths reserved
This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

## Summary
"""

This example demonstrates the usage of the tudatpy `porkchop` module to determine an optimal launch window (departure and arrival date) for a **low-thrust** Earth-Mars transfer mission.
By default, the porkchop module uses a Lambert arc to compute the $\Delta V$ required to depart from the departure body (Earth in this case) and be captured by the target body (in this case Mars).
Users can provide a custom function to calculate the $\Delta V$ required for any given transfer. This can be done by supplying a `callable` (a function) to the `porkchop` function via the argument

    function_to_calculate_delta_v

In this example, this option will be used to choose (make a preliminary choice, that is) the optimal departure and arrival date of a low-thrust transfer from the Earth to Mars.
"""

## Import statements
"""

The required import statements are made here, starting with standard imports (`os`, `pickle` from the Python Standard Library), followed by tudatpy imports.
"""

# General imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Tudatpy imports
import tudatpy
from tudatpy import constants
from tudatpy import numerical_simulation
from tudatpy.interface import spice
from tudatpy.astro.time_conversion import DateTime
from tudatpy.trajectory_design import shape_based_thrust
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.trajectory_design.porkchop import porkchop, plot_porkchop

# Tudatpy data processing utilities
from tudatpy.numerical_simulation.propagation import create_dependent_variable_dictionary
from tudatpy.util import result2array


## Environment setup
"""

The simulation environment is set up here: the standard Spice kernels are loaded, the origin of the global frame is defined, and all necessary bodies are created. 

"""

# Load spice kernels
spice.load_standard_kernels( )

# Define global frame orientation
global_frame_orientation = 'ECLIPJ2000'

# Create bodies
bodies_to_create = ['Sun', 'Venus', 'Earth', 'Moon', 'Mars', 'Jupiter', 'Saturn']
global_frame_origin = 'Sun'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create vehicle object and add it to the existing system of bodies
vehicle_mass = 4.0E3
specific_impulse = 3000.0
body_settings.add_empty_settings("Vehicle")

body_settings.get("Vehicle").constant_mass = vehicle_mass

# Create rotation model settings
rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time : np.array([1,0,0] ), global_frame_orientation, 'VehicleFixed' )
body_settings.get("Vehicle").rotation_model_settings = rotation_model_settings

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle thrust settings        
thrust_magnitude_settings = (
propagation_setup.thrust.custom_thrust_magnitude_fixed_isp( lambda time : 0.0, specific_impulse ) )
environment_setup.add_engine_model(
    'Vehicle', 'LowThrustEngine', thrust_magnitude_settings, bodies )


## Shape-based low-thrust trajectory optimization
"""

Define the necessary parameters of the low-thrust trajectory:

- Number of revolutions around the Sun
- Free parameters for radial shaping functions
- Free parameters for normal shaping functions
- Free parameters for axial shaping functions

"""

number_of_revolutions = 2

radial_velocity_shaping_free_coefficients = [
    2471.19649906354,
    4207.587982407276
]
normal_velocity_shaping_free_coefficients = [
    -5594.040587888714,
    8748.139268525232,
]
axial_velocity_shaping_free_coefficients = [
    -3449.838496679572,
    0.0
]


### Velocity shaping functions
"""

Define a factory function to obtain the radial velocity shaping functions

"""

def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the radial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the radial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[3:5]
    return (radial_velocity_shaping_functions,
            free_coefficients)


# Define a factory function to obtain the normal velocity shaping functions

def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the normal velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the normal velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[5:7]
    return (normal_velocity_shaping_functions,
            free_coefficients)


# Define a factory function to obtain the axial velocity shaping functions

def get_axial_velocity_shaping_functions(trajectory_parameters: list,
                                         frequency: float,
                                         scale_factor: float,
                                         time_of_flight: float,
                                         number_of_revolutions: int) -> tuple:
    """
    Retrieves the axial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list[ float ]
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun.

    Returns
    -------
    tuple
        A tuple composed by two lists: the axial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    exponent = 4.0
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    # Set free parameters
    free_coefficients = trajectory_parameters[7:9]
    return (axial_velocity_shaping_functions,
            free_coefficients)


### Low-thrust Trajectory Optimization solution
"""

Define a function to obtain the LTTO solution
"""

def create_hodographic_trajectory(
        trajectory_parameters: list,
        bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
        departure_body: str,
        target_body: str,
        central_body: str) \
        -> tudatpy.trajectory_design.transfer_trajectory.TransferTrajectory:
    """
    It creates and returns the hodographic shaping object, based on the trajectory parameters.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    hodographic_shaping_object : tudatpy.trajectory_design.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    """

    # Time settings
    initial_time = trajectory_parameters[0] * constants.JULIAN_DAY
    time_of_flight = trajectory_parameters[1] * constants.JULIAN_DAY
    final_time = initial_time + time_of_flight
    
    # Number of revolutions
    number_of_revolutions = int(trajectory_parameters[2])
    
    # Compute relevant frequency and scale factor for shaping functions
    frequency = 2.0 * np.pi / time_of_flight
    scale_factor = 1.0 / time_of_flight
    
    # Retrieve shaping functions and free parameters
    radial_velocity_shaping_functions, radial_free_coefficients = get_radial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    normal_velocity_shaping_functions, normal_free_coefficients = get_normal_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    axial_velocity_shaping_functions, axial_free_coefficients = get_axial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)

    # Create settings for transfer trajectory (zero excess velocity on departure and arrival)
    hodographic_leg_settings = transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_shaping_functions,
        normal_velocity_shaping_functions,
        axial_velocity_shaping_functions )
    node_settings = list()
    node_settings.append( transfer_trajectory.departure_node( 1.0E8, 0.0 ) )
    node_settings.append( transfer_trajectory.capture_node( 1.0E8, 0.0 ) )

    # Create and return transfer trajectory
    trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies, [hodographic_leg_settings], node_settings, [departure_body, target_body], central_body )

    # Extract node times
    node_times = list( )
    node_times.append( initial_time )
    node_times.append( final_time )

    #transfer_trajectory.print_parameter_definitions( [hodographic_leg_settings], node_settings )
    hodograph_free_parameters = trajectory_parameters[2:9]

    # Depart and arrive with 0 excess velocity
    node_parameters = list()
    node_parameters.append( np.zeros([3,1]))
    node_parameters.append( np.zeros([3,1]))

    # Update trajectory to given times, node settings, and hodograph parameters
    trajectory_object.evaluate( node_times, [hodograph_free_parameters], node_parameters )

    return trajectory_object


# Create function to obtain transfer $\Delta V$

def hodographic_low_thrust_trajectory_delta_v(
        bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
        departure_body: str,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float,
        central_body: str = 'Sun') \
    -> list[tudatpy.trajectory_design.transfer_trajectory.TransferTrajectory, float]:
    """
    Function to calculate the required ΔV of an Earth-Mars transfer

    Parameters
    ----------
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        The system of bodies containing the celestial bodies involved in the transfer.
    departure_body : str
        The name of the departure celestial body.
    target_body : str
        The name of the target celestial body.
    departure_epoch : float
        The departure epoch in seconds since J2000.
    arrival_epoch : float
        The arrival epoch in seconds since J2000.
    central_body : str, optional
        The name of the central celestial body (default is 'Sun').

    Returns
    -------
    [tudatpy.trajectory_design.transfer_trajectory.TransferTrajectory, float]
        A tuple containing the transfer trajectory object and the required ΔV.
    """
    
    # The entries of the vector 'trajectory_parameters' contains the following:
    # * Entry 0: Departure time (from Earth's center-of-mass) in Julian days since J2000
    # * Entry 1: Time-of-flight from Earth's center-of-mass to Mars' center-of-mass, in Julian days
    # * Entry 2: Number of revolutions around the Sun
    # * Entry 3,4: Free parameters for radial shaping functions
    # * Entry 5,6: Free parameters for normal shaping functions
    # * Entry 7,8: Free parameters for axial shaping functions
    
    trajectory_parameters = [
        departure_epoch / constants.JULIAN_DAY,
        (arrival_epoch - departure_epoch) / constants.JULIAN_DAY,
        number_of_revolutions,
        *radial_velocity_shaping_free_coefficients,
        *normal_velocity_shaping_free_coefficients,
        *axial_velocity_shaping_free_coefficients
    ]

    hodographic_shaping_object = create_hodographic_trajectory(
        trajectory_parameters,
        bodies,
        departure_body,
        target_body,
        central_body)

    # Retrieve delta V
    ΔV = hodographic_shaping_object.delta_v

    return ΔV


## Porkchop Plots
"""

The departure and target bodies and the time window for the transfer are then defined using tudatpy `astro.time_conversion.DateTime` objects.
"""

departure_body = 'Earth'
target_body = 'Mars'

earliest_departure_time = DateTime(2016,  9,   1)
latest_departure_time   = DateTime(2017,  7,   1)

earliest_arrival_time   = DateTime(2019, 11,   1)
latest_arrival_time     = DateTime(2021,  9,   1)


# To ensure the porkchop plot is rendered with good resolution, the time resolution of the plot is defined as 0.5% of the smallest time window (either the arrival or the departure window):

time_window_percentage = 0.5
time_resolution = time_resolution = min(
        latest_departure_time.epoch() - earliest_departure_time.epoch(),
        latest_arrival_time.epoch()   - earliest_arrival_time.epoch()
) / constants.JULIAN_DAY * time_window_percentage / 100


# Generating a high-resolution plot may be time-consuming: reusing saved data might be desirable; we proceed to ask the user whether to reuse saved data or generate the plot from scratch.

# File
data_file = 'porkchop.pkl'

# Whether to recalculate the porkchop plot or use saved data
RECALCULATE_delta_v = input(
    '\n    Recalculate ΔV for porkchop plot? [y/N] '
).strip().lower() == 'y'
print()


# Lastly, we call the `porkchop` function, which will calculate the $\Delta V$ required at each departure-arrival coordinate and display the plot, giving us
#  
# - The optimal departure-arrival date combination
# - The constant time-of-flight isochrones
# - And more

if not os.path.isfile(data_file) or RECALCULATE_delta_v:
    # Regenerate plot
    [departure_epochs, arrival_epochs, ΔV] = porkchop(
        bodies,
        departure_body,
        target_body,
        earliest_departure_time,
        latest_departure_time,
        earliest_arrival_time,
        latest_arrival_time,
        time_resolution,
        function_to_calculate_delta_v=hodographic_low_thrust_trajectory_delta_v
    )
    # Save data
    pickle.dump(
        [departure_epochs, arrival_epochs, ΔV],
        open(data_file, 'wb')
    )
else:
    # Read saved data
    [departure_epochs, arrival_epochs, ΔV] = pickle.load(
        open(data_file, 'rb')
    )
    # Plot saved data
    plot_porkchop(
        departure_body   = departure_body,
        target_body      = target_body,
        departure_epochs = departure_epochs, 
        arrival_epochs   = arrival_epochs, 
        delta_v          = ΔV,
        threshold        = 60
    )


### Variations
"""

The Tudat `porkchop` module allows us to

- Save the $\Delta V$ porkchop returned by `porkchop` and plot it again without recalculating with the `plot_porkchop` function
- Plot $\Delta V$ (default) or C3 (specific energy), as well as choose whether to plot departure and arrival $\Delta V$ together as the total $\Delta V$ required for the transfer (default), or separately (in those cases in which the manoeuvre is performed in two burns, one at departure and one at arrival to the target planet).

Let's make use of `plot_porkchop` to see all four combinations!
"""

cases = [
    {'C3': False, 'total': True,  'threshold': 80,   'filename': 'figures/Δ_tot.png'},
    {'C3': True,  'total': True,  'threshold': 3000, 'filename': 'figures/C3_tot.png'}
]

for case in cases:
    plot_porkchop(
        departure_body   = departure_body,
        target_body      = target_body,
        departure_epochs = departure_epochs, 
        arrival_epochs   = arrival_epochs, 
        delta_v          = ΔV,
        save             = False,
        **case
    )


# Verification
"""

Interestingly, discontinuities appear in the $\Delta V$/$C_3$ porkchop. To ensure that the porkchop can be trusted -that it is possible to choose a transfer window based on our porkchop-, we will proceed to investigate two transfers:

- The (approximately) minimum $\Delta V$ transfer: `2016-11-16`-`2020-6-11`
- A high $\Delta V$ transfer in the dark red region of the porkchop: `2016-10-22`-`2021-01-21`
"""

## Trajectory visualization
"""

Provided with a transfer window, the following function will obtain the shape-based low thrust trajectory from the Earth to Mars, numerically propagate a trajectory using a low-thrust thrust model for our spacecraft, and plot the: 

- Cartesian coordinates of the spacecraft, as a function of time, both for the analytical and integrated trajectory
- The Cartesian coordinates as a function of time of the Earth and Mars
- The thrust acceleration on the spacecraft as a function of time
- And a 3D plot showing the complete manoeuvre
"""

def inspect_low_thrust_trajectory(
        departure_date: DateTime,
        arrival_date: DateTime
    ):
    """
    This function has the following sections:

    1. Define transfer parameters
    2. Obtain low-thrust shape-based semi-analytical trajectory
    3. Create vehicle thrust settings
    4. Create termination settings
    5. Propagator settings
    6. Integrator settings
    7. Propagate dynamics
    8. Process simulation output
    9. Retrieve ephemeris of astronomical bodies
    10. Plot trajectory 
    """

    ###########################################################################
    # DEFINE TRANSFER PARAMETERS ##############################################
    ###########################################################################

    trajectory_parameters = [
        departure_date.epoch() / constants.JULIAN_DAY,
        (arrival_date.epoch() - departure_date.epoch()) / constants.JULIAN_DAY,
        number_of_revolutions,
        *radial_velocity_shaping_free_coefficients,
        *normal_velocity_shaping_free_coefficients,
        *axial_velocity_shaping_free_coefficients
    ]

    # Fixed parameters
    minimum_mars_distance = 5.0E7
    # Time since 'departure from Earth CoM' at which propagation starts (and similar
    # for arrival time)
    time_buffer = 30.0 * constants.JULIAN_DAY

    # Propagation time settings
    initial_propagation_time = departure_date.epoch() + time_buffer
    final_propagation_time = arrival_date.epoch() - time_buffer

    ###########################################################################
    # OBTAIN LOW-THRUST SHAPE-BASED SEMI-ANALYTICAL TRAJECTORY ################
    ###########################################################################

    # Create problem without propagating
    hodographic_shaping_object = create_hodographic_trajectory(
        trajectory_parameters, 
        bodies,
        'Earth',
        'Mars',
        'Sun'
    )

    # Retrieves analytical results and write them to a file
    analytical_trajectory = lambda n: result2array(hodographic_shaping_object.states_along_trajectory(n))[:, 1:]

    # Report transfer ΔV
    print(f'{hodographic_shaping_object.delta_v/1000:.2f} km/s')

    ###########################################################################
    # CREATE TERMINATION SETTINGS #############################################
    ###########################################################################

    # Time settings
    time_termination_settings = propagation_setup.propagator.time_termination(
        final_propagation_time,
        terminate_exactly_on_final_condition=False)
    # Altitude
    relative_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
        limit_value=minimum_mars_distance,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False)
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 relative_distance_termination_settings]
    # Create termination settings object
    termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                           fulfill_single_condition=True)
    
    # Retrieve dependent variables to save
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
        propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
        propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.thrust_acceleration_type,'Vehicle','Vehicle')
    ]

    ###########################################################################
    # PROPAGATOR SETTINGS #####################################################
    ###########################################################################

    current_propagator = propagation_setup.propagator.unified_state_model_quaternions

    # Define propagation settings
        # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']

    # Update vehicle rotation model and thrust magnitude model
    transfer_trajectory.set_low_thrust_acceleration( hodographic_shaping_object.legs[ 0 ], bodies, 'Vehicle', 'LowThrustEngine' )
    
    # Define accelerations acting on capsule
    acceleration_settings_on_vehicle = {
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [propagation_setup.acceleration.thrust_from_engine('LowThrustEngine')]
    }

    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = hodographic_shaping_object.legs[ 0 ].state_along_trajectory( initial_propagation_time )

    # Create propagation settings for the translational dynamics
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_propagation_time,
        None,
        termination_settings,
        current_propagator,
        output_variables=dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_vehicle,
                                                                 acceleration_models)
    # Create mass propagator settings
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([vehicle_mass]),
                                                                 initial_propagation_time,
                                                                 None,
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object for translational dynamics and mass
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 None,
                                                                 initial_propagation_time,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)
    
    ###########################################################################
    # INTEGRATOR SETTINGS #####################################################
    ###########################################################################

    # Create integrator settings
    current_tolerance = 10.0 ** (-10.0)
    # Create integrator settings
    integrator = propagation_setup.integrator
    # Define fixed step size
    step_size = constants.JULIAN_DAY
    # Here (epsilon, inf) are set as respectively min and max step sizes
    # also note that the relative and absolute tolerances are the same value
    integrator_settings = integrator.runge_kutta_variable_step_size(
        step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87,
        step_size,
        step_size,
        current_tolerance,
        current_tolerance)
    propagator_settings.integrator_settings = integrator_settings

    ###########################################################################
    # PROPAGATE DYNAMICS ######################################################
    ###########################################################################
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    ###########################################################################
    # PROCESS SIMULATION OUTPUT ###############################################
    ###########################################################################
    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.propagation_results.state_history

    # Create dependent variable dictionary
    dv_dict = create_dependent_variable_dictionary(dynamics_simulator)

    # Create state history array
    state_history_array = result2array(state_history)

    # Retrieve propagation buffer time
    buffer = time_buffer / constants.JULIAN_DAY

    # Retrieve time vector
    t = state_history_array[:, 0] / constants.JULIAN_DAY
    t = (t - t[0]) + buffer

    # Retrieve analytical time vector
    t_analytical = np.arange(0, max(t)+buffer+1, step_size/constants.JULIAN_DAY)

    # Retrieve x, y and z position history
    x = state_history_array[:, 1]
    y = state_history_array[:, 2]
    z = state_history_array[:, 3]

    # Retrieve analytical trajectory
    x_analytical = analytical_trajectory(len(t_analytical))[:, 0]
    y_analytical = analytical_trajectory(len(t_analytical))[:, 1]
    z_analytical = analytical_trajectory(len(t_analytical))[:, 2]

    # Retrieve thrust magnitude
    thrust_magnitude = dv_dict.asarray('Single acceleration norm of type thrust , acting on Vehicle')

    ###########################################################################
    # RETRIEVE EPHEMERIS OF ASTRONOMICAL BODIES ###############################
    ###########################################################################

    retrieve_ephemeris = lambda body: np.vstack([spice.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name="SSB",
        reference_frame_name="ECLIPJ2000",
        aberration_corrections="NONE",
        ephemeris_time=epoch_in_days * constants.JULIAN_DAY
    )[:3] for epoch_in_days in np.linspace(trajectory_parameters[0], trajectory_parameters[0] + trajectory_parameters[1], len(t_analytical))])

    # Retrieve trajectory of the Earth
    earth_trajectory = retrieve_ephemeris('Earth')
    earth_x = earth_trajectory[:, 0]
    earth_y = earth_trajectory[:, 1]
    earth_z = earth_trajectory[:, 2]

    # Retrieve trajectory of mars
    mars_trajectory = retrieve_ephemeris('Mars')
    mars_x = mars_trajectory[:, 0]
    mars_y = mars_trajectory[:, 1]
    mars_z = mars_trajectory[:, 2]

    ###########################################################################
    # PLOT TRAJECTORY #########################################################
    ###########################################################################

    # Create figure
    fig = plt.figure(figsize=(15, 7))

    # Create 3 subplots for x, y, z coordinates
    ax1 = plt.subplot2grid((4, 2), (0, 0))
    ax2 = plt.subplot2grid((4, 2), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((4, 2), (2, 0), sharex=ax1)
    ax4 = plt.subplot2grid((4, 2), (3, 0), sharex=ax1)

    # Plot analytical and numerical coordinates
    ax1.plot(t_analytical, x_analytical, label='Analytical')
    ax1.plot(t, x, label='Numerical')
    ax1.plot(t_analytical, earth_x, label='Earth', color='#78c4ff', zorder=1, alpha=0.6)
    ax1.plot(t_analytical, mars_x, label='Mars', color='#f04848', zorder=1, alpha=0.6)
    ax1.set_ylabel('X [km]')

    ax2.plot(t_analytical, y_analytical)
    ax2.plot(t, y)
    ax2.plot(t_analytical, earth_y, color='#78c4ff', zorder=1, alpha=0.6)
    ax2.plot(t_analytical, mars_y, color='#f04848', zorder=1, alpha=0.6)
    ax2.set_ylabel('Y [km]')

    ax3.plot(t_analytical, z_analytical)
    ax3.plot(t, z)
    ax3.plot(t_analytical, earth_z, color='#78c4ff', zorder=1, alpha=0.6)
    ax3.plot(t_analytical, mars_z, color='#f04848', zorder=1, alpha=0.6)
    ax3.set_ylabel('Z [km]')

    ax4.plot(t, thrust_magnitude / vehicle_mass, color='green', label='Thrust acceleration')
    ax4.set_ylabel('$||a_T||$ [$m/s^2$]')
    ax4.set_xlabel('Time [days]')

    # Legend
    fig.legend(loc='lower center', ncol=5)

    # Create a 3D subplot for the trajectories
    ax5 = plt.subplot2grid((3, 2), (0, 1), rowspan=3, projection='3d')
    ax5.view_init(elev=32, azim=-66)

    # Plot Earth trajectory
    ax5.plot(earth_x, earth_y, earth_z, color='#78c4ff')
    ax5.scatter(earth_x[0], earth_y[0], earth_z[0], label='Earth, $t_0$', marker='$\Lambda$', s=100, color='#1f9eff')

    # Plot Mars trajectory
    ax5.plot(mars_x, mars_y, mars_z, color='#f04848')
    ax5.scatter(mars_x[-1], mars_y[-1], mars_z[-1], label='Mars, $t_f$', marker='$\Gamma$', s=100, color='#c93232')

    # Plot the 3D trajectories
    ax5.plot(x_analytical, y_analytical, z_analytical)
    ax5.plot(x, y, z)

    # Set labels
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    # Show the legend
    ax5.legend()

    # Grid
    for ax in [ax1, ax2, ax3, ax4]: ax.grid(
        True,
        which='major',
        color='grey',
        linestyle='--'
    )

    # Layout
    plt.subplots_adjust(
        top=0.95,
        bottom=0.127,
        left=0.043,
        right=0.99,
        hspace=0.335,
        wspace=0.0
    )

    # Show the plot
    plt.show()


## Observations and conclusions
"""

Notice that there is a change in the number of revolutions around the Sun between the two transfers: the low $\Delta V$ transfer has close to 3 revolutions around the Sun, while the high $\Delta V$ transfer greatly "stretches" its second revolution around the Sun to reach Mars. This is expected as the chosen number of revolutions (2) allows trajectories with between 2 and 3 revolutions around the Sun: thus, a discontinuity appears as trajectories approach 3 revolutions, and are from that point on forced to use 2; this is the case for the high $\Delta V$ trajectory.
The thrust acceleration of the high $\Delta V$ trajectory is 2 orders of magnitude higher than that for the low $\Delta V$ trajectory. This is expected as the orbit of the spacecraft in the high $\Delta V$ trajectory is distorted considerably more than in the low $\Delta V$ case.

The discontinuities observed in the $\Delta V$ porkchops are explained by a discontinuous change in the number of revolutions required for the low-thrust transfer. The porkchops generated are thus reasonable, and can be used to identify a preliminary transfer window for our low-thrust spacecraft, as well as obtain a preliminary estimate of the $\Delta V$ required for the mission.
Next steps in mission design should be the optimization of the shaping function parameters (`radial_velocity_shaping_free_coefficients`, `normal_velocity_shaping_free_coefficients` and `axial_velocity_shaping_free_coefficients`) and number of revolutions around the Sun, which were fixed in this example, analyzing disturbing perturbations along the trajectory, analyzing the impact of navigation uncertainties on the trajectory, et cétera.

"""

# **Low $\Delta V$ trajectory**

inspect_low_thrust_trajectory(
    DateTime(2016,11,16),
    DateTime(2020,6,11)
)


# **High $\Delta V$ trajectory**

inspect_low_thrust_trajectory(
    DateTime(2016,10,22),
    DateTime(2021,1,21)
)

