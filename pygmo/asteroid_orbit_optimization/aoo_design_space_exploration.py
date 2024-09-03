# Asteroid orbit optimization with PyGMO - Design Space Exploration
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution  and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.

"""

## Context
"""
This tutorial is the second part of the Asteroid Orbit Optimization example. **This page reuses the** [Custom environment](https://tudat-space.readthedocs.io/en/latest/_src_getting_started/_src_examples/notebooks/pygmo/asteroid_orbit_optimization/aoo_custom_environment.html) **part of the example, without the explanation, after which a Design Space Exploration (DSE) is done**. The DSE is collection of methods with which an optimization problem can be analyzed, and better understood, without having to execute an optimization or test randomly.

"""

## Problem definition
"""

The 4 design variables are:

- initial values of the semi-major axis.
- initial eccentricity.
- initial inclination.
- initial longitude of the ascending node.
 
The 2 objectives are:

- good coverage (maximizing the mean value of the absolute longitude w.r.t. Itokawa over the full propagation).
- good resolution (the mean value of the distance should be minimized).
 
The constraints are set on the altitude: all the sets of design variables leading to an orbit.

"""

#### NOTE
"""
It is assumed that the reader of this tutorial is already familiar with the content of [this basic PyGMO tutorial](https://tudat-space.readthedocs.io/en/latest/_src_advanced_topics/optimization_pygmo.html). The full PyGMO documentation is available [on this website](https://esa.github.io/pygmo2/index.html). Be careful to read the
correct the documentation webpage (there is also a similar one for previous yet now outdated versions [here](https://esa.github.io/pygmo/index.html); as you can see, they can easily be confused).
PyGMO is the Python counterpart of [PAGMO](https://esa.github.io/pagmo2/index.html).
"""

## Import statements
"""
# Load standard modules
import os
import numpy as np
Uncomment the following to make plots interactive
%matplotlib widget
from matplotlib import pyplot as plt
from itertools import combinations as comb
"""


# Load tudatpy modules
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.astro import element_conversion
from tudatpy.astro import frame_conversion
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.util as util

# Load pygmo library
import pygmo as pg

current_dir = os.path.abspath('')


## Creation of Custom Environment
"""
"""

### Itokawa rotation settings
"""
"""

def get_itokawa_rotation_settings(itokawa_body_frame_name):
    # Definition of initial Itokawa orientation conditions through the pole orientation
    pole_declination = np.deg2rad(-66.30)     # Declination
    pole_right_ascension = np.deg2rad(90.53)  # Right ascension
    meridian_at_epoch = 0.0                   # Meridian

    # Define initial Itokawa orientation in inertial frame (equatorial plane)
    initial_orientation_j2000 = frame_conversion.inertial_to_body_fixed_rotation_matrix(
        pole_declination, pole_right_ascension, meridian_at_epoch)
    
    # Get initial Itokawa orientation in inertial frame but in the Ecliptic plane
    initial_orientation_eclipj2000 = np.matmul(spice.compute_rotation_matrix_between_frames(
        "J2000", "ECLIPJ2000", 0.0), initial_orientation_j2000)

    # Manually check the results, if desired
    check_results = False
    if check_results:
        np.set_printoptions(precision=100)
        print(initial_orientation_j2000)
    
        print(initial_orientation_eclipj2000)

    # Compute rotation rate
    rotation_rate = np.deg2rad(712.143) / constants.JULIAN_DAY

    # Set up rotational model for Itokawa with constant angular velocity
    return environment_setup.rotation_model.simple(
        "ECLIPJ2000", itokawa_body_frame_name, initial_orientation_eclipj2000, 0.0, rotation_rate)


### Itokawa ephemeris settings
"""
def get_itokawa_ephemeris_settings(sun_gravitational_parameter):
    Define Itokawa initial Kepler elements
    itokawa_kepler_elements = np.array([
        1.324118017407799 * constants.ASTRONOMICAL_UNIT,
        0.2801166461882852,
        np.deg2rad(1.621303507642802),
        np.deg2rad(162.8147699851312),
        np.deg2rad(69.0803904880264),
        np.deg2rad(187.6327516838828)])
    
    Convert mean anomaly to true anomaly
    itokawa_kepler_elements[5] = element_conversion.mean_to_true_anomaly(
        eccentricity=itokawa_kepler_elements[1],
        mean_anomaly=itokawa_kepler_elements[5])
    
    Get epoch of initial Kepler elements (in Julian Days)
    kepler_elements_reference_julian_day = 2459000.5
    
    Sets new reference epoch for Itokawa ephemerides (different from J2000)
    kepler_elements_reference_epoch = (kepler_elements_reference_julian_day - constants.JULIAN_DAY_ON_J2000) \
                                      * constants.JULIAN_DAY
    
    Sets the ephemeris model
    return environment_setup.ephemeris.keplerian(
        itokawa_kepler_elements,
        kepler_elements_reference_epoch,
        sun_gravitational_parameter,
        "Sun",
        "ECLIPJ2000")
"""


### Itokawa gravity field settings
"""
def get_itokawa_gravity_field_settings(itokawa_body_fixed_frame, itokawa_radius):
    itokawa_gravitational_parameter = 2.36
    normalized_cosine_coefficients = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.145216, 0.0, 0.219420, 0.0, 0.0],
        [0.036115, -0.028139, -0.046894, 0.069022, 0.0],
        [0.087852, 0.034069, -0.123263, -0.030673, 0.150282]])
    normalized_sine_coefficients = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.006137, -0.046894, 0.033976, 0.0],
        [0.0, 0.004870, 0.000098, -0.015026, 0.011627]])
    return environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter=itokawa_gravitational_parameter,
        reference_radius=itokawa_radius,
        normalized_cosine_coefficients=normalized_cosine_coefficients,
        normalized_sine_coefficients=normalized_sine_coefficients,
        associated_reference_frame=itokawa_body_fixed_frame)
"""


### Itokawa shape settings
"""
def get_itokawa_shape_settings(itokawa_radius):
    Creates spherical shape settings
    return environment_setup.shape.spherical(itokawa_radius)
"""


### Simulation bodies
"""
def create_simulation_bodies(itokawa_radius):
"""

    ##CELESTIAL BODIES ###
"""
    Define Itokawa body frame name
    itokawa_body_frame_name = "Itokawa_Frame"
"""

    # Create default body settings for selected celestial bodies
    bodies_to_create = ["Sun", "Earth", "Jupiter", "Saturn", "Mars"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as
    # global frame origin and orientation. This environment will only be valid
    # in the indicated time range [simulation_start_epoch --- simulation_end_epoch]
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        "SSB",
        "ECLIPJ2000")

    # Add Itokawa body
    body_settings.add_empty_settings("Itokawa")

    # Adds Itokawa settings
    # Gravity field
    body_settings.get("Itokawa").gravity_field_settings = get_itokawa_gravity_field_settings(itokawa_body_frame_name,
                                                                        itokawa_radius)
    # Rotational model
    body_settings.get("Itokawa").rotation_model_settings = get_itokawa_rotation_settings(itokawa_body_frame_name)
    # Ephemeris
    body_settings.get("Itokawa").ephemeris_settings = get_itokawa_ephemeris_settings(
        spice.get_body_gravitational_parameter( 'Sun') )
    # Shape (spherical)
    body_settings.get("Itokawa").shape_settings = get_itokawa_shape_settings(itokawa_radius)
    # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ### VEHICLE BODY ###
    # Create vehicle object
    body_settings.add_empty_settings("Spacecraft")
    body_settings.get("Spacecraft").constant_mass = 400

    # Create radiation pressure settings
    reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
    radiation_pressure_coefficient = 1.2
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Itokawa"]
    vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

    # Add the radiation pressure interface to the body settings
    body_settings.get("Spacecraft").radiation_pressure_target_settings = vehicle_target_settings

    # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


### Acceleration models
"""
def get_acceleration_models(bodies_to_propagate, central_bodies, bodies):
    Define accelerations acting on Spacecraft
    accelerations_settings_spacecraft = dict(
        Sun =     [ propagation_setup.acceleration.radiation_pressure(),
                    propagation_setup.acceleration.point_mass_gravity() ],
        Itokawa = [ propagation_setup.acceleration.spherical_harmonic_gravity(3, 3) ],
        Jupiter = [ propagation_setup.acceleration.point_mass_gravity() ],
        Saturn =  [ propagation_setup.acceleration.point_mass_gravity() ],
        Mars =    [ propagation_setup.acceleration.point_mass_gravity() ],
        Earth =   [ propagation_setup.acceleration.point_mass_gravity() ]
    )
"""

    # Create global accelerations settings dictionary
    acceleration_settings = {"Spacecraft": accelerations_settings_spacecraft}

    # Create acceleration models
    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)


### Termination settings
"""
def get_termination_settings(mission_initial_time, 
                             mission_duration,
                             minimum_distance_from_com,
                             maximum_distance_from_com):
    Mission duration
    time_termination_settings = propagation_setup.propagator.time_termination(
        mission_initial_time + mission_duration,
        terminate_exactly_on_final_condition=False
    )
    Upper altitude
    upper_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Spacecraft', 'Itokawa'),
        limit_value=maximum_distance_from_com,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )
    Lower altitude
    lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Spacecraft', 'Itokawa'),
        limit_value=minimum_distance_from_com,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
"""

    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 upper_altitude_termination_settings,
                                 lower_altitude_termination_settings]

    return propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                           fulfill_single_condition=True)


### Dependent variables to save
"""
def get_dependent_variables_to_save():
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.central_body_fixed_spherical_position(
            "Spacecraft", "Itokawa"
        )
    ]
    return dependent_variables_to_save
"""


## Optimisation problem formulation 
"""
class AsteroidOrbitProblem:
    
    def __init__(self,
                 bodies,
                 integrator_settings,
                 propagator_settings,
                 mission_initial_time,
                 mission_duration,
                 design_variable_lower_boundaries,
                 design_variable_upper_boundaries):
        
        Sets input arguments as lambda function attributes
        NOTE: this is done so that the class is "pickable", i.e., can be serialized by pygmo
        self.bodies_function = lambda: bodies
        self.integrator_settings_function = lambda: integrator_settings
        self.propagator_settings_function = lambda: propagator_settings
        
        Initialize empty dynamics simulator
        self.dynamics_simulator_function = lambda: None
        
        Set other input arguments as regular attributes
        self.mission_initial_time = mission_initial_time
        self.mission_duration = mission_duration
        self.mission_final_time = mission_initial_time + mission_duration
        self.design_variable_lower_boundaries = design_variable_lower_boundaries
        self.design_variable_upper_boundaries = design_variable_upper_boundaries
"""

    def get_bounds(self):
        return (list(self.design_variable_lower_boundaries), list(self.design_variable_upper_boundaries))

    def get_nobj(self):
        return 2

    def fitness(self,
                orbit_parameters):
        # Retrieves system of bodies
        current_bodies = self.bodies_function()
        
        # Retrieves Itokawa gravitational parameter
        itokawa_gravitational_parameter = current_bodies.get("Itokawa").gravitational_parameter
        
        # Reset the initial state from the design variable vector
        new_initial_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=itokawa_gravitational_parameter,
            semi_major_axis=orbit_parameters[0],
            eccentricity=orbit_parameters[1],
            inclination=np.deg2rad(orbit_parameters[2]),
            argument_of_periapsis=np.deg2rad(235.7),
            longitude_of_ascending_node=np.deg2rad(orbit_parameters[3]),
            true_anomaly=np.deg2rad(139.87))
        
        # Retrieves propagator settings object
        propagator_settings = self.propagator_settings_function()
        
        # Retrieves integrator settings object
        integrator_settings = self.integrator_settings_function()
        
        # Reset the initial state
        propagator_settings.initial_states = new_initial_state

        # Propagate orbit
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(current_bodies,
                                                                        propagator_settings)
        
        # Update dynamics simulator function
        self.dynamics_simulator_function = lambda: dynamics_simulator

        # Retrieve dependent variable history
        dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
        dependent_variables_list = np.vstack(list(dependent_variables.values()))
        
        # Retrieve distance
        distance = dependent_variables_list[:, 0]
        # Retrieve latitude
        latitudes = dependent_variables_list[:, 1]
        
        # Compute mean latitude
        mean_latitude = np.mean(np.absolute(latitudes))
        # Computes fitness as mean latitude
        current_fitness = 1.0 / mean_latitude

        # Exaggerate fitness value if the spacecraft has broken out of the selected distance range
        current_penalty = 0.0
        if (max(dynamics_simulator.propagation_results.dependent_variable_history.keys()) < self.mission_final_time):
            current_penalty = 1.0E2

        return [current_fitness + current_penalty, np.mean(distance) + current_penalty * 1.0E3]

    def get_last_run_dynamics_simulator(self):
        return self.dynamics_simulator_function()


### Setup orbital simulation
"""

"""

#### Simulation settings
"""
# Load spice kernels
spice.load_standard_kernels()
"""

# Set simulation start and end epochs
mission_initial_time = 0.0
mission_duration = 5.0 * constants.JULIAN_DAY

# Define Itokawa radius
itokawa_radius = 161.915

# Set altitude termination conditions
minimum_distance_from_com = 150.0 + itokawa_radius
maximum_distance_from_com = 5.0E3 + itokawa_radius

# Set boundaries on the design variables
design_variable_lb = (300, 0.0, 0.0, 0.0)
design_variable_ub = (2000, 0.3, 180, 360)

# Create simulation bodies
bodies = create_simulation_bodies(itokawa_radius)

# Define bodies to propagate and central bodies
bodies_to_propagate = ["Spacecraft"]
central_bodies = ["Itokawa"]

# Create acceleration models
acceleration_models = get_acceleration_models(bodies_to_propagate, central_bodies, bodies)


#### Dependent variables, termination settings, and orbit parameters
"""
# Define list of dependent variables to save
dependent_variables_to_save = get_dependent_variables_to_save()
"""

# Create propagation settings
termination_settings = get_termination_settings(
    mission_initial_time, mission_duration, minimum_distance_from_com, maximum_distance_from_com)

orbit_parameters = [1.20940330e+03, 2.61526215e-01, 7.53126558e+01, 2.60280587e+02]


#### Integrator and Propagator settings
"""
# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    initial_time_step=1.0,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
    minimum_step_size=1.0E-6,
    maximum_step_size=constants.JULIAN_DAY,
    relative_error_tolerance=1.0E-8,
    absolute_error_tolerance=1.0E-8)
"""

# Get current propagator, and define translational state propagation settings
propagator = propagation_setup.propagator.cowell
# Define propagation settings
initial_state = np.zeros(6)
propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                         acceleration_models,
                                                                         bodies_to_propagate,
                                                                         initial_state,
                                                                         mission_initial_time,
                                                                         integrator_settings,
                                                                         termination_settings,
                                                                         propagator,
                                                                         dependent_variables_to_save)


## Design Space Exploration
"""

**From here on out the example is new compared to the** [Custom environment](https://tudat-space.readthedocs.io/en/latest/_src_getting_started/_src_examples/notebooks/pygmo/asteroid_orbit_optimization/aoo_custom_environment.html) **part of the example.**

Now that the simulation has been setup, the problem can actually be run and explored. While one could jump into the optimisation immediately, not much is known yet about the specific problem at hand. A design space exploration is done prior to the optimisation in order to better understand the behaviour of the system. The goal is to figure out and observe the link between the design space and the objective space. Numerous methods for exploring the design space are possible, a list of the implemented methods can be seen below. This selection covers various kinds of analysis, ranging from simple and brainless, to systematic and focussed. 

- Monte Carlo Analysis
- Fractional Factorial Design
- Factorial Design
"""

### Monte Carlo Analysis
"""
Starting with the method that requires the least amount of thinking; a Monte Carlo Analysis. By varying input parameters randomly, and propagating many trajectories, one can discover trends; how the semi-major axis influences the mean latitude objective, for example. The difficulty arises in that the results are not conclusive; design variables can be coupled by definition.
"""

#### Variable Definitions
"""

A number of variables have to be defined. The number of runs per design variable, this quantity is a trade-off between resolution of your results and time spent. The seed is defined for reproducibility of the results. A number of arrays are defined for saving the data relevant for post-processing. 
"""

no_of_runs = 500
random_seed = 42

np.random.seed(random_seed)

orbit_param_names = ['Semi-major Axis', 'Eccentricity', 'Inclination', 'Longitude of the Node']
mean_latitude_all_param = np.zeros((no_of_runs, len(orbit_param_names)))
mean_distance_all_param = np.zeros((no_of_runs, len(orbit_param_names)))
max_distance = np.zeros((no_of_runs, len(orbit_param_names)))
min_distance = np.zeros((no_of_runs, len(orbit_param_names)))
constraint_values = np.zeros((no_of_runs, len(orbit_param_names)))
parameters = np.zeros((no_of_runs, len(orbit_param_names)))


#### Monte Carlo loop
"""

The Monte Carlo variation is made with two nested loops; one for the various orbit parameters that will be changed, and one for each run. As explained before, only one parameter is changed per run, so for each parameter, a set of random numbers is produced equal to the number of simulations. This new combination is throughput into the fitness function of the `AsteroidOrbitProblem` class. Regarding that `AsteroidOrbitProblem` class, for the sake of consistency, the (UDP) Problem class from PyGMO is used. This class is by no means necessary for running the analysis. After the fitness is evaluated, a number of relevant quantities are saved to the previously defined arrays.
"""

for i in range(len(orbit_parameters)): 

    #print('Monte Carlo design variable :', orbit_param_names[i])
    parameter_all_runs = np.random.uniform(design_variable_lb[i], design_variable_ub[i], no_of_runs)
    parameters[:, i] = parameter_all_runs


    for j in range(no_of_runs):
        #print('Monte Carlo Run :', str(j))

        initial_state[i] = parameter_all_runs[j]
        orbit_parameters[i] = parameter_all_runs[j]

        # Create Asteroid Orbit Problem object
        current_asteroid_orbit_problem = AsteroidOrbitProblem(bodies,
                                                                  integrator_settings,
                                                                  propagator_settings,
                                                                  mission_initial_time,
                                                                  mission_duration,
                                                                  design_variable_lb,
                                                                  design_variable_ub
                                                                  )

        # Update thrust settings and evaluate fitness
        current_asteroid_orbit_problem.fitness(orbit_parameters)

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.state_history
        dependent_variable_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.dependent_variable_history

        # Get the number of function evaluations (for comparison of different integrators)
        dynamics_simulator = current_asteroid_orbit_problem.get_last_run_dynamics_simulator()
        function_evaluation_dict = dynamics_simulator.propagation_results.cumulative_number_of_function_evaluations_history
        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]

        dependent_variables_list = np.vstack(list(dependent_variable_history.values()))
        # Retrieve distance
        distance = dependent_variables_list[:, 0]
        # Retrieve latitude
        latitudes = dependent_variables_list[:, 1]
        # Compute mean latitude
        mean_latitude = np.mean(np.absolute(latitudes))
        # Compute mean distance
        mean_distance = np.mean(distance)  

        # Objectives
        mean_latitude_all_param[j, i] = mean_latitude
        mean_distance_all_param[j, i] = mean_distance

        max_dist = np.max(distance)
        min_dist = np.min(distance)

        
        dist_to_max = maximum_distance_from_com - max_dist
        dist_to_min = minimum_distance_from_com - min_dist

        if np.abs(dist_to_max) > np.abs(dist_to_min):
            constraint_val = np.abs(dist_to_min)
            maxmin = 'min'
        else:
            constraint_val = np.abs(dist_to_max)
            maxmin = 'max'
        constraint_values[j, i] = constraint_val


####  Monte Carlo Post-processing
"""
A few dictionaries are defined for labelling purposes and the parameters are scattered against the objective values. A color map is used to indicate how close the solutions are to the distance constraint value. Many results are made, but only the semi-major axis variation data is plotted. Remove `break` in the nested loop below to obtain all results.
"""

# Create dictionaries defining the design variables
design_variable_names = {0: 'Semi-major axis [m]',
                           1: 'Eccentricity',
                           2: 'Inclination [deg]',
                           3: 'Longitude of the node [deg]'}
design_variable_range = {0: [800.0, 1300.0],
                           1: [0.10, 0.17],
                           2: [90.0, 95.0],
                           3: [250.0, 270.0]}
design_variable_symbols = {0: r'$a$',
                             1: r'$e$',
                             2: r'$i$',
                             3: r'$\Omega$'}
design_variable_units = {0: r' m',
                           1: r' ',
                           2: r' deg',
                           3: r' deg'}


obj_arrays = [mean_latitude_all_param, mean_distance_all_param]
objective_names = ['Latitude', 'Distance']
for obj in range(2): #number of objectives

    for i in range(len(orbit_param_names)):
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.suptitle('Monte Carlo - one-by-one - Objective: %s - Scaling: Constrained Distance'%(objective_names[obj]))
        for ax_index, ax in enumerate(axs.flatten()):
            cs = ax.scatter(parameters[:, ax_index], obj_arrays[obj][:, ax_index], s=2, c=constraint_values[:,i])
            cbar = fig.colorbar(cs, ax=ax)
            cbar.ax.set_ylabel('Distance constraint value')
            ax.set_ylabel('%s [rad]'%(objective_names[obj]))
            ax.set_xlabel(design_variable_names[ax_index])
        
        #For more verbose results, remove the 'break' below.
        break


### Fractional Factorial Design
"""
The Fractional Factorial Design (FFD) method has a number of pros and cons relative to the Monte Carlo method. The concept is based on orthogonality of a design matrix, with which you can extract information efficiently without running a ton of simulations. In other words, a selection of corners of the design space hypercube are explored. The advantage of the orthogonal array, based on Latin Squares, is that it is computationally very light, thereby of course sacrificing knowledge about your design space. The information per run is high with FFD.

"""

#### Orthogonal Array
"""

A function, `get_orthogonal_array()`,  is used that calculates the orthogonal array depending on the number of levels (2 or 3) and number of factors (design variables) that any specific problem has. The algorithm is based on the Latin Square and the array is systematically built from there. The content of this array can sometimes be confusing, but it is quite straightforward; the rows represent experiments, the columns represent the factors, and the entries represent the discretized value of the factor — in the two-level case -1 becomes the minimum bound and 1 becomes the maximum bound. If you print the array that rolls out, you can get a feel for the structure of the method and the reason why it is efficient.

"""

#### Fractional Factorial Design Loop 
"""

The FFD module is similar to that of the Monte Carlo in that there are two loops including one that changes the design variables and one that loops over the various runs. Within the two loops the `AsteroidOrbitProblem` is defined again and the objective values are extracted after evaluating the fitness function.

The difference is in how the orbit parameters are assigned. Instead of creating vectors of random numbers, the values in the orthogonal array, -1 and 1 for 2-level analysis, map to minimum and maximum orbit parameter values respectively.
"""

no_of_factors = 4 # This leads to an 8x7 orthogonal array. Meaning 7 contribution percentages will be given in the ANOVA.
no_of_levels = 2

FFD_array = util.get_orthogonal_array(no_of_factors, no_of_levels)
mean_dependent_variables_list = np.zeros((len(FFD_array), 2)) # the second argument is the number of objectives

for i in range(len(FFD_array)):
    for j in range(len(orbit_parameters)):
        
        if FFD_array[i,j] == -1:
            orbit_parameters[j] = design_variable_lb[j]
        else: # if two level orthogonal array
            orbit_parameters[j] = design_variable_ub[j]

    # Create Asteroid Orbit Problem object
    current_asteroid_orbit_problem = AsteroidOrbitProblem(bodies,
                                                  integrator_settings,
                                                  propagator_settings,
                                                  mission_initial_time,
                                                  mission_duration,
                                                  design_variable_lb,
                                                  design_variable_ub
                                                  )
    # Update orbital parameters and evaluate fitness
    current_asteroid_orbit_problem.fitness(orbit_parameters)

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.state_history
    dependent_variable_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.dependent_variable_history

    dependent_variables_list = np.vstack(list(dependent_variable_history.values()))
    # Retrieve distance
    distance = dependent_variables_list[:, 0]
    # Retrieve latitude
    latitudes = dependent_variables_list[:, 1]
    # Compute mean latitude
    mean_latitude = np.mean(np.absolute(latitudes))
    # Compute mean distance
    mean_distance = np.mean(distance)

    mean_dependent_variables_list[i, 0] = mean_distance
    mean_dependent_variables_list[i, 1] = mean_latitude


#### Post-processing FFD
"""
As not many runs are done, plotting any data is not sensible. An Analysis of Variance (ANOVA) can be done to determine percentage contributions of each parameter. For example, one would find that the eccentricity—one of the design variables—has a x% contribution to the distance objective.

This ANOVA analysis can also be done on the Factorial Design method discussed below, and so in the interest of space it is only applied there.
"""

### Factorial Design
"""

Factorial design (FD) is another systematic approach to exploring the design space. It can be very useful as far fewer assumptions are made about the results; FD is complete in that all corners—and potentially intermediate points—of the hypercube are tested. Whereas with FFD an orthogonal array was created with Latin Squares, here the array is built using Yates algorithm. Some information can be found [here](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35i.htm).

"""

#### Yates Array
"""
The Yates array is similar to the orthogonal array in that it is orthogonal, and the rows, columns, and entries correspond to the same things (experiments, factors, and discretised values, respectively). The Yates array has significantly more rows, because it is complete, as mentioned before.

"""

#### FD loop
"""
The orbit parameters are now assigned using the input from Yates array discussed before. Otherwise the structure is the same as Monte Carlo and FFD.
"""

no_of_levels = 7 # make 7 for the response surfaces
no_of_factors = len(orbit_parameters)

#Function to create Yates array
yates_array = util.get_yates_array(no_of_factors, no_of_levels)
no_of_sims = len(yates_array)

design_variable_arr = np.zeros((no_of_levels, no_of_factors))
for par in range(no_of_factors):
    design_variable_arr[:, par] = np.linspace(design_variable_lb[par], design_variable_ub[par], no_of_levels, endpoint=True)

param_arr = np.zeros((len(yates_array),len(orbit_parameters)))
objective_arr = np.zeros((len(yates_array),2))

mean_distances = np.zeros(no_of_sims) 
mean_latitudes = np.zeros(no_of_sims)
for i in range(len(yates_array)): # Run through yates array
    level_combination = yates_array[i, :]
    for it, j in enumerate(level_combination): #Run through the row of levels from 0 to no_of_levels
        if no_of_levels == 7:
            orbit_parameters[it] = design_variable_arr[j+3, it]
        else:
            if j == -1:
                orbit_parameters[it] = design_variable_arr[0, it]
            if j == 1:
                orbit_parameters[it] = design_variable_arr[1, it]
    orbit_param_copy = orbit_parameters.copy()
    param_arr[i,:] = orbit_param_copy


    # Create Asteroid Orbit Problem object
    current_asteroid_orbit_problem = AsteroidOrbitProblem(bodies,
                                                  integrator_settings,
                                                  propagator_settings,
                                                  mission_initial_time,
                                                  mission_duration,
                                                  design_variable_lb,
                                                  design_variable_ub
                                                  )
    # Update orbital parameters and evaluate fitness
    current_asteroid_orbit_problem.fitness(orbit_parameters)

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.state_history
    dependent_variable_history = current_asteroid_orbit_problem.get_last_run_dynamics_simulator().propagation_results.dependent_variable_history

    dependent_variables_list = np.vstack(list(dependent_variable_history.values()))
    # Retrieve distance
    distance = dependent_variables_list[:, 0]
    # Retrieve latitude
    latitudes = dependent_variables_list[:, 1]
    # Compute mean latitude
    mean_latitude = np.mean(np.absolute(latitudes))
    # Compute mean distance
    mean_distance = np.mean(distance)

    objective_arr[i,:] = np.array([mean_distance, mean_latitude])
    mean_distances[i] = mean_distance
    mean_latitudes[i] = mean_latitude


#### Anova Analysis
"""

Now that yates array has been created and the objective values have been obtained, these two pieces of data can be combined to calculate what the contribution is of a certain variable or interaction to an objective, using an ANOVA analysis. Individual, linear, and quadratic effects are can be taken into account when determining the contributions.
"""

# Anova analysis
i, ij, ijk, err = util.anova_analysis(mean_distances, #can also be mean_latitudes
            yates_array,
            no_of_factors,
            no_of_levels,
            level_of_interactions=2)


#### ANOVA Results
"""
In the tables below, the individual, linear, and quadratic contributions to the distance objective can be found in percentages, which follows from the anova_analysis function. NOTE: These results were made with the 2-level yates array, because the  interaction columns are calculated with -1 and 1. This doesn't work with 7 levels.

|    |  Semi-major Axis  |Eccentricity|Inclination|Longitude of the Node|
|----|----|----|----|----|
|Individual Contribution [%]|99.7|0.101|0.037|1.38e-4|

|    |Sma-Ecc|Sma-Inc|Sma-Lon|Ecc-Inc|Ecc-Lon|Inc-Lon|
|----|----|----|----|----|----|----|
|Linear Interaction [%]|6.06e-5|0.037|1.38e-4|2.51e-2|3.55e-4|2.28e-4|

|    |  Sma-Ecc-Inc  |Sma-Ecc-Lon|Sma-Inc-Lon|Ecc-Inc-Lon|
|----|----|----|----|----|
|Quadratic Contribution [%]|2.51e-2|3.55e-4|2.28e-4|6.42e-5|
"""

#### Response Surface Post-processing
"""
With factorial design, a response surface can also be plotted. These surfaces are generally only useful if the resolution is higher than 2, as you can only see linear trends with two levels. The problem can easily be too large (too many design variables) to run the problem with 7 levels, but for this problem it is doable. The following results are thus created by setting the no_of_levels to 7.

As plotting all the data obtained with the FD is rather verbose, and probably not the best way, each combination of two variables is plotted for the trajectories where the other two parameters are at their minimum—the 0th index.

A few lists are created for labelling, and the iterators for each response surface plot are set to 0. The 6 combinations with their conditions as explained before are implemented, after which the 49 points are plotted (`no_of_levels**2`).

"""

it1, it2, it3, it4, it5, it6 = 0, 0, 0, 0, 0, 0
combi_list = ['sma_ecc', 'sma_inc', 'sma_lon', 'ecc_inc', 'ecc_lon', 'inc_lon']
xlabel_list = ['Semi-major Axis [m]','Semi-major Axis [m]', 'Semi-major Axis [m]', 'Eccentricity [-]', 'Eccentricity [-]', 'Inclination [rad]']
ylabel_list = ['Eccentricity [-]', 'Inclination [rad]', 'Longitude of the Node [rad]', 'Inclination [rad]', 'Longitude of the Node [rad]', 'Longitude of the Node [rad]']
title_list = ['inc = 0 & lon = 0', 'ecc = 0 & lon = 0', 'ecc = 0 & inc = 0', 'sma = 300 & lon = 0', 'sma = 300 & inc = 0', 'sma = 300 & ecc = 0']

objectives = {}
params = {}
for i in combi_list:
    objectives[i] = np.zeros((no_of_levels**2, 2))
    params[i] = np.zeros((no_of_levels**2, 2))
for i in range(len(param_arr)):
    if param_arr[i, 2] == 0 and param_arr[i, 3] == 0:
        objectives['sma_ecc'][it1, :] = objective_arr[i, :] 
        params['sma_ecc'][it1, :] = param_arr[i, [0, 1]]
        it1 += 1
    if param_arr[i, 1] == 0 and param_arr[i, 3] == 0:
        objectives['sma_inc'][it2, :] = objective_arr[i, :] 
        params['sma_inc'][it2, :] = param_arr[i, [0, 2]]
        it2 += 1
    if param_arr[i, 1] == 0 and param_arr[i, 2] == 0:
        objectives['sma_lon'][it3, :] = objective_arr[i, :] 
        params['sma_lon'][it3, :] = param_arr[i, [0, 3]]
        it3 += 1
    if param_arr[i, 0] == 300 and param_arr[i, 3] == 0:
        objectives['ecc_inc'][it4, :] = objective_arr[i, :] 
        params['ecc_inc'][it4, :] = param_arr[i, [1, 2]]
        it4 += 1
    if param_arr[i, 0] == 300 and param_arr[i, 2] == 0:
        objectives['ecc_lon'][it5, :] = objective_arr[i, :] 
        params['ecc_lon'][it5, :] = param_arr[i, [1, 3]]
        it5 += 1
    if param_arr[i, 0] == 300 and param_arr[i, 1] == 0:
        objectives['inc_lon'][it6, :] = objective_arr[i, :] 
        params['inc_lon'][it6, :] = param_arr[i, [1, 3]]
        it6 += 1

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Response surfaces each combination - 0th level for other parameters', fontweight='bold', y=0.95)
for i, combi in enumerate(combi_list):
    ax = fig.add_subplot(2, 3, 1 + i, projection='3d')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    cmap = ax.plot_surface(params[combi][:, 0].reshape(no_of_levels,
        no_of_levels), params[combi][:, 1].reshape(no_of_levels,
            no_of_levels), objectives[combi][:, 1].reshape(no_of_levels,
                no_of_levels), cmap=plt.get_cmap('copper'))

    ax.set_xlabel(xlabel_list[i], labelpad=5)
    ax.set_ylabel(ylabel_list[i], labelpad=5)
    ax.set_zlabel('Mean Latitude [rad]', labelpad=10)
    ax.set_title('%s - %s '%(combi, title_list[i]), y=1.0, pad=10)
    ax.view_init(10, -140)

