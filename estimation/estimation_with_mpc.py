# %% [markdown]
# Initial state estimation with Minor Planet Center Observations
"""
Copyright (c) 2010-2023, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""


## Context
"""
This example highlights a simple orbit estimation routine using real, angular observation data from the  [Minor Planet Center](https://www.minorplanetcenter.net/) (MPC). We will estimate the initial state of [Ceres](https://en.wikipedia.org/wiki/Ceres_(dwarf_planet)) the largest body in the asteroid belt. We will use the Tudat BatchMPC interface to retrieve and process the data. For a more in depth explanation of this interface we recommend first checking out the [Retrieving observation data from the Minor Planet Centre](https://docs.tudat.space/en/latest/_src_getting_started/_src_examples/notebooks/estimation/retrieving_mpc_observation_data.html) example.
"""

# %% [markdown]
## Import statements
"""
"""

# %%
# Tudat imports for propagation and estimation
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# import MPC interface
from tudatpy.data.mpc import BatchMPC

# other useful modules
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# %% [markdown]
## Preparing the environment and observations
"""
"""


### Loading Spice Kernels.
"""
We use SPICE kernels to retrieve the ephemerides the planets as well as to verify our results for Ceres. The ephemerides for Ceres and other asteroids are loaded in with the standard kernels.
"""

# %%
# SPICE KERNELS
spice.load_standard_kernels()

# %% [markdown]
### Retrieving the observations
"""
We retrieve the observation data using the BatchMPC interface. By default all observation data is retrieved, even the first observation from Piazzi in 1801. We filter to only include data between January 2018 and July 2023.
"""

# %%
codes = [1]

batch = BatchMPC()
batch.get_observations(codes)
batch.filter(
    epoch_start=datetime.datetime(2018, 1, 1),
    epoch_end=datetime.datetime(2023, 7, 1),
)

batch.summary()

# %% [markdown]
# Our batch includes many observations from space telescopes, lets take a closer look at that data.

# %%
print("Summary of space telescopes in batch:")
print(batch.observatories_table(only_space_telescopes=True))
obs_by_WISE = batch.table.query("observatory == 'C51'").loc[:, ["number", "epochUTC", "RA", "DEC"]].iloc[[0, -1]]

print("\nInitial and Final Observations by WISE:")
print(obs_by_WISE)

# %% [markdown]
# While the observations from WISE appear to be useful, including them requires setting up the dynamics for the WISE spacecraft which is too advanced for this tutorial and its observations will be excluded automatically later on in this example. The observations can also be filtered out explicitly by excluding the observatories with the .filter() method. Note that all the observations are given in an angular format, Right Ascension (RA) and Declination (DEC).

# %% [markdown]
### Set up the environment
"""
We now set up the environment, including the bodies to use, the reference frame and frame origin. The epherides for all major planets as well as the Earth's Moon are included retrieved using spice. For our frame origin we use the Solar System Barycentre. The data from MPC is presented in the J2000 reference frame, currently BatchMPC does not support conversion to other reference frames and as such we match it in our environment. 

BatchMPC will automatically generate the body object for Ceres, but we still need to specify the bodies to propagate and their central bodies. We can retrieve the list from the BatchMPC object.
"""

# %%
# List the bodies for our environment
bodies_to_create = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Moon",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

# define the frame origin and orientation. 
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Retrieve Ceres' body name from BatchMPC and set its centre
bodies_to_propagate = batch.MPC_objects
central_bodies = ["SSB"]

# %% [markdown]
### Convert the observations to Tudat
"""
Now that our system of bodies is ready we can retrieve the observation collection and links from the observations batch. With the links we just need to define the observation settings, this is where you would also add corrections. For the purpose of this example, we will keep it simple and use the plain angular position settings, which can process observations with Right Ascension and Declination. We can also retrieve the times for the first and final observations from the batch object.
"""

# %%
# Transform the MPC observations into a tudat compatible format.
observation_collection, links = batch.to_tudat(bodies=bodies)

# set create angular_position settings for each link in the list.
observation_settings_list = list()
for link in list(links.values()):
    observation_settings_list.append(observation.angular_position(link))

# Retrieve the first and final observation epochs
epoch_start = batch.epoch_start
epoch_end = batch.epoch_end

# %% [markdown]
### Creating the acceleration settings
"""
Ceres will be propagated and as such we need to define the settings of the forces acting on it. We will include point mass gravity accelerations for each of the bodies defined before, as well as Schhwarzschild relativistic corrections for the sun. With these accelerations we can generate our acceleration model for the propagation. A more realistic acceleration model will yield better results but this is outside the scope of this example.
"""

# %%
# Define accelerations
accelerations = {
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.relativistic_correction(use_schwarzschild = True),
    ],
    "Mercury": [propagation_setup.acceleration.point_mass_gravity()],
    "Venus": [propagation_setup.acceleration.point_mass_gravity()],
    "Earth": [propagation_setup.acceleration.point_mass_gravity()],
    "Moon": [propagation_setup.acceleration.point_mass_gravity()],
    "Mars": [propagation_setup.acceleration.point_mass_gravity()],
    "Jupiter": [propagation_setup.acceleration.point_mass_gravity()],
    "Saturn": [propagation_setup.acceleration.point_mass_gravity()],
    "Uranus": [propagation_setup.acceleration.point_mass_gravity()],
    "Neptune": [propagation_setup.acceleration.point_mass_gravity()],
}

# Set up the accelerations settings for each body, in this case only Ceres
acceleration_settings = {}
for body in batch.MPC_objects:
    acceleration_settings[str(body)] = accelerations

# create the acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# %% [markdown]
### Retrieving an initial guess for Ceres' position
"""
We use the SPICE ephemeris to retrieve a 'benchmark' initial state for Ceres at the final epoch. We can also use this initial state as our initial guess for the estimation. To ensure a working estimation we add a random uniform offset of +/- 1 million kilometers for the position and 100 m/s for the velocity. We take the final time to retrieve the 'latest' position.
"""

# %%
# benchmark state for later comparison retrieved from SPICE
initial_states = spice.get_body_cartesian_state_at_epoch("Ceres", "SSB", "J2000", "NONE", epoch_end)

# Add random offset for initial guess
np.random.seed = 1

initial_guess = initial_states.copy()
initial_guess[0:3] += (2*np.random.rand(3) - 1) * (1e6 * 1000)
initial_guess[3:6] += (2*np.random.rand(3) - 1) * 100

print("Error between initial state and initial guess:")
print(initial_guess - initial_states)

# %% [markdown]
### Finalising the propagation setup
"""
In this example we set up the propagation to run in reverse. We set the timestep to minus 20 hours. For the integrator we use the fixed timestep RKF-7(8) setting our initial time to the time of the batch's final observation. We then set the termination to stop at the time of the batch's oldest observation. These two settings are then the final pieces to create our propagation settings. 
"""

# %%
# timestep of 20 hours
dt = -(20 * 3600)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    epoch_end, dt, propagation_setup.integrator.rkf_78, dt, dt, 1.0, 1.0
)

# Terminate at the time of oldest observation
termination_condition = propagation_setup.propagator.time_termination(epoch_start)


# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=central_bodies,
    acceleration_models=acceleration_models,
    bodies_to_integrate=bodies_to_propagate,
    initial_states=initial_guess,
    initial_time=epoch_end,
    integrator_settings=integrator_settings,
    termination_settings=termination_condition,
)

# %% [markdown]
## Setting Up the estimation
"""
With the observation collection, the environment and propagations settings ready we can now begin setting up our estimation. 

In this example we will simply estimate the position of Ceres and as such only include an initial states parameter.
"""

# %%
# Setup parameters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(
    propagator_settings, bodies
)

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(
    parameter_settings, bodies, propagator_settings
)

# %% [markdown]
# The `Estimator` object collects the environment, observation settings and propagation settings. We also create an `EstimationInput` object and provide it our observation collection retrieved from `.to_tudat()` as well as setting the maximum iteration steps to 6.

# %%
# Set up the estimator
estimator = numerical_simulation.Estimator(
    bodies=bodies,
    estimated_parameters=parameters_to_estimate,
    observation_settings=observation_settings_list,
    propagator_settings=propagator_settings,
    integrate_on_creation=True, 
)

# provide the observation collection as input, and limit number of iterations for estimation.
pod_input = estimation.EstimationInput(
    observations_and_times=observation_collection,
    convergence_checker=estimation.estimation_convergence_checker(
        maximum_iterations=4,
    )
)

# Set methodological options
pod_input.define_estimation_settings(reintegrate_variational_equations=True)

# %% [markdown]
## Performing the estimation and analysing results
"""

With everything set up we can now perform the estimation. 
"""

# %%
# Perform the estimation
pod_output = estimator.perform_estimation(pod_input)

# %% [markdown]
# The estimator appears to converge within 4 steps. Lets check how close our initial and final guesses are compared to the benchmark state.

# %%
# retrieve the final estimated state.
results_final = pod_output.parameter_history[:, -1]

vector_error_initial = (np.array(initial_guess) - initial_states)[0:3]
error_magnitude_initial = np.sqrt(np.square(vector_error_initial).sum())/1000

vector_error_final = (np.array(results_final) - initial_states)[0:3]
error_magnitude_final = np.sqrt(np.square(vector_error_final).sum())/1000

print(f"Ceres initial radial error: {round(error_magnitude_initial, 2)} km")
print(f"Ceres final radial error: {round(error_magnitude_final, 2)} km")

# %% [markdown]
# Lets also take a look at the residuals over time:

# %%
residual_history = pod_output.residual_history

fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=False)

# a little trick to retrieve the UTC times from the batch, table retrieves all observations in pandas format
times = (
    batch.table.query("observatory != 'C51'")
    .set_index("epochJ2000secondsTDB")
    .loc[observation_collection.concatenated_times]
    .epochUTC
    .sort_values()
    .tolist()
)

for idx, ax in enumerate(fig.get_axes()):
    ax.grid()
    ax.scatter(times, residual_history[:, idx], marker="+", s=60)
    ax.set_ylabel("Observation Residual [rad]")
    ax.set_title("Iteration " + str(idx + 1))

plt.tight_layout()

axs[1, 0].set_xlabel("Year")
axs[1, 1].set_xlabel("Year")

plt.show()

# %% [markdown]
# Lets also look at the orbit error over time compared to spice:

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

spice_states = []
estimation_states = []

# buffer to avoid interpolation at the edges of the estimated ephemeris
buffer = 86400 * 2

times = np.linspace(epoch_start + buffer, epoch_end - buffer, 1000)
times_plot = times/(86400*365.25) + 2000 # approximate
for time in times:
    # from spice
    state_spice = spice.get_body_cartesian_state_at_epoch("Ceres", central_bodies[0], "J2000", "NONE", time)
    spice_states.append(state_spice)
    
    # from estimation
    state_est = bodies.get("1").ephemeris.cartesian_state(time)
    estimation_states.append(state_est)

# Error in kilometers
error = (np.array(spice_states) - np.array(estimation_states)) / 1000

ax.plot(times_plot, error[:, 0], label="x")
ax.plot(times_plot, error[:, 1], label="y")
ax.plot(times_plot, error[:, 2], label="z")

ax.grid()
ax.legend(ncol=3)

plt.tight_layout()

ax.set_ylabel("Carthesian Error [km]")
ax.set_xlabel("Year")

plt.show()

# %% [markdown]
# The final result is within the radius of Ceres, but there is clearly plenty of room for improvement in both the dynamical model and the estimation settings. Consider for example adding weights and biases on observations and links as well as improved integrator settings and perturbations.


