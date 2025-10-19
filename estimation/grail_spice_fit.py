# GRAIL - Fitting various models of the GRAIL spacecraft's dynamics to the reference spice trajectory
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""


## Context
"""
"""

# Load required standard modules
import multiprocessing as mp
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy.data import grail_mass_level_0_file_reader
from tudatpy.interface import spice
from tudatpy.astro import time_representation
from tudatpy import util

from tudatpy.dynamics import environment_setup
from tudatpy.dynamics.environment_setup import radiation_pressure
from tudatpy.dynamics import propagation_setup, parameters_setup
from tudatpy import estimation
from tudatpy.estimation import estimation_analysis

from datetime import datetime

# Import GRAIL examples functions
from grail_examples_functions import (
    get_grail_files,
    get_grail_panel_geometry,
    get_rsw_state_difference,
)


# This function tests various setups (both in terms of dynamical model and set of parameters to estimate) to fit GRAIL's orbit to
# its reference spice trajectory, over arcs of one day. This is done by sampling the spice ephemeris and generating ideal "position"
# observables. We then fit GRAIL's dynamical model to such observables, using different set of estimated parameters.
#
# The "inputs" variable used as input argument is a list with eleven entries:
#   1- the index of the current run (the run_spice_fit function being run in parallel on several cores in this example)
#   2- the date for the day-long arc under consideration
#   3- the index of the setup to consider for the current run (defines both GRAIL's dynamical model and the list of estimated parameters)
#   4- the clock file to be loaded
#   5- the list of orientation kernels to be loaded
#   6- the GRAIL manoeuvres file to be loaded
#   7- the list of GRAIL trajectory files to be loaded
#   8- the GRAIL reference frames file to be loaded
#   9- the lunar orientation kernel to be loaded
#   10- the lunar reference frame kernel to be loaded
#   11- the output files directory


def run_spice_fit(inputs):

    # Unpack various input arguments
    input_index = inputs[0]

    # Retrieve the current date as string
    date_string = inputs[1].strftime("%m/%d/%Y").replace("/", "")

    # Convert the datetime object defining the current date to a Tudat Time variable.
    # A time buffer of 10 min is added to ensure that the GRAIL orientation kernel fully covers the time interval of interest,
    # without interpolation errors in case the current kernel starts exactly on the day under consideration.
    start_time = (
        time_representation.DateTime.from_python_datetime(inputs[1]).to_epoch_time_object() + 600.0
    )
    end_time = start_time + 86400.0

    # Retrieve index of the setup to consider (this defines both the model to be used to propagate GRAIL's dynamics
    # and the list of estimated parameters in the fit)
    index_setup = inputs[2]

    # Retrieve lists of relevant kernels and input files to load (clock and orientation kernels for GRAIL, manoeuvres file,
    # GRAIL trajectory files, GRAIL reference frames file, lunar orientation kernels, and lunar reference frame kernel)
    clock_file = inputs[3]
    grail_orientation_files = inputs[4]
    manoeuvre_file = inputs[5]
    trajectory_files = inputs[6]
    grail_ref_frames_file = inputs[7]
    lunar_orientation_file = inputs[8]
    lunar_ref_frame_file = inputs[9]

    # Retrieve output folder
    output_folder = inputs[10]

    # Redirect the outputs of this run to a file names grail_spice_fit_output_DATE_setup_x.dat, with x the setup index and
    # 'DATE' the date of interest written as MMDDYYYYY
    with util.redirect_std(
        output_folder
        + "grail_spice_fit_output_"
        + date_string
        + "_setup_"
        + str(index_setup)
        + ".dat",
        True,
        True,
    ):

        print("index_setup", index_setup)
        print("date_string", date_string)

        filename_suffix = date_string + "_setup_" + str(index_setup)

        ### ------------------------------------------------------------------------------------------
        ### LOAD ALL REQUESTED KERNELS AND FILES
        ### ------------------------------------------------------------------------------------------

        # Load standard spice kernels
        spice.load_standard_kernels()

        # Load specific Moon kernels
        spice.load_kernel(lunar_ref_frame_file)
        spice.load_kernel(lunar_orientation_file)

        # Load GRAIL frame definition file (useful for spacecraft-fixed frames definition)
        spice.load_kernel(grail_ref_frames_file)

        # Load GRAIL orientation kernels (over the entire relevant time period).
        for orientation_file in grail_orientation_files:
            spice.load_kernel(orientation_file)

        # Load GRAIL clock file
        spice.load_kernel(clock_file)

        # Load GRAIL trajectory kernel
        for trajectory_file in trajectory_files:
            spice.load_kernel(trajectory_file)

        ### ------------------------------------------------------------------------------------------
        ### CREATE DYNAMICAL ENVIRONMENT
        ### ------------------------------------------------------------------------------------------

        # Create default body settings for celestial bodies
        bodies_to_create = [
            "Earth",
            "Sun",
            "Mercury",
            "Venus",
            "Mars",
            "Jupiter",
            "Saturn",
            "Moon",
        ]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create,
            start_time.to_float(),
            end_time.to_float(),
            global_frame_origin,
            global_frame_orientation,
        )

        # Modify default rotation and gravity field settings for the Moon
        body_settings.get("Moon").rotation_model_settings = (
            environment_setup.rotation_model.spice(
                global_frame_orientation, "MOON_PA_DE440", "MOON_PA_DE440"
            )
        )
        body_settings.get("Moon").gravity_field_settings = (
            environment_setup.gravity_field.predefined_spherical_harmonic(
                environment_setup.gravity_field.gggrx1200, 500
            )
        )
        body_settings.get("Moon").gravity_field_settings.associated_reference_frame = (
            "MOON_PA_DE440"
        )

        # Define gravity field variations for the tides on the Moon
        moon_gravity_field_variations = list()
        moon_gravity_field_variations.append(
            environment_setup.gravity_field_variation.solid_body_tide(
                "Earth", 0.02405, 2
            )
        )
        moon_gravity_field_variations.append(
            environment_setup.gravity_field_variation.solid_body_tide("Sun", 0.02405, 2)
        )
        body_settings.get("Moon").gravity_field_variation_settings = (
            moon_gravity_field_variations
        )
        body_settings.get("Moon").ephemeris_settings.frame_origin = "Earth"

        # Add Moon radiation properties
        moon_surface_radiosity_models = [
            radiation_pressure.thermal_emission_angle_based_radiosity(
                95.0, 385.0, 0.95, "Sun"
            ),
            radiation_pressure.variable_albedo_surface_radiosity(
                radiation_pressure.predefined_spherical_harmonic_surface_property_distribution(
                    radiation_pressure.albedo_dlam1
                ),
                "Sun",
            ),
        ]
        body_settings.get("Moon").radiation_source_settings = (
            radiation_pressure.panelled_extended_radiation_source(
                moon_surface_radiosity_models, [6, 12]
            )
        )

        # Create empty settings for the GRAIL spacecraft
        spacecraft_name = "GRAIL-A"
        spacecraft_central_body = "Moon"
        body_settings.add_empty_settings(spacecraft_name)
        body_settings.get(spacecraft_name).constant_mass = 150

        # Define translational ephemeris from SPICE
        body_settings.get(spacecraft_name).ephemeris_settings = (
            environment_setup.ephemeris.interpolated_spice(
                start_time.to_float(),
                end_time.to_float(),
                10.0,
                spacecraft_central_body,
                global_frame_orientation,
            )
        )

        # Define rotational ephemeris from SPICE
        body_settings.get(spacecraft_name).rotation_model_settings = (
            environment_setup.rotation_model.spice(
                global_frame_orientation, spacecraft_name + "_SPACECRAFT", ""
            )
        )

        # Define GRAIL panel geometry, which will be used for the panel radiation pressure model
        body_settings.get(spacecraft_name).vehicle_shape_settings = (
            get_grail_panel_geometry()
        )

        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Add radiation pressure target models for GRAIL (both cannonball and complete panel models)
        occulting_bodies = dict()
        occulting_bodies["Sun"] = ["Moon"]
        environment_setup.add_radiation_pressure_target_model(
            bodies,
            spacecraft_name,
            radiation_pressure.cannonball_radiation_target(5, 1.5, occulting_bodies),
        )
        environment_setup.add_radiation_pressure_target_model(
            bodies,
            spacecraft_name,
            radiation_pressure.panelled_radiation_target(occulting_bodies),
        )

        ### ------------------------------------------------------------------------------------------
        ### RETRIEVE GRAIL MANOEUVRES EPOCHS
        ### ------------------------------------------------------------------------------------------

        # Load the times at which the spacecraft underwent a manoeuvre from GRAIL's manoeuvres file
        manoeuvres_times = grail_mass_level_0_file_reader(manoeuvre_file)

        # Store the manoeuvres epochs if they occur within the time interval under consideration
        relevant_manoeuvres = []
        for manoeuvre_time in manoeuvres_times:
            if (
                manoeuvre_time >= start_time.to_float()
                and manoeuvre_time <= end_time.to_float()
            ):
                print("manoeuvre detected")
                relevant_manoeuvres.append(manoeuvre_time)

        ### ------------------------------------------------------------------------------------------
        ### DEFINE ACCELERATION MODELS
        ### ------------------------------------------------------------------------------------------

        # Define two different lists of accelerations acting on GRAIL (a simplified dynamical model and a more complete one).
        # The model actually used to propagate GRAIL's dynamics depends on the current setup index.

        simple_accelerations_settings_spacecraft = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure(
                    environment_setup.radiation_pressure.cannonball_target
                )
            ],
            Earth=[propagation_setup.acceleration.point_mass_gravity()],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(10, 10),
                propagation_setup.acceleration.radiation_pressure(
                    environment_setup.radiation_pressure.cannonball_target
                ),
            ],
        )

        complete_accelerations_settings_spacecraft = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure(
                    environment_setup.radiation_pressure.paneled_target
                ),
                propagation_setup.acceleration.point_mass_gravity(),
            ],
            Earth=[propagation_setup.acceleration.point_mass_gravity()],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(256, 256),
                propagation_setup.acceleration.radiation_pressure(
                    environment_setup.radiation_pressure.cannonball_target
                ),
            ],
            Mars=[propagation_setup.acceleration.point_mass_gravity()],
            Venus=[propagation_setup.acceleration.point_mass_gravity()],
            Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        )

        # Add manoeuvres if necessary
        if len(relevant_manoeuvres) > 0:
            complete_accelerations_settings_spacecraft[spacecraft_name] = [
                propagation_setup.acceleration.quasi_impulsive_shots_acceleration(
                    relevant_manoeuvres, [np.zeros((3, 1))], 3600.0, 60.0
                )
            ]

        # Create accelerations settings (for index_setup = 0, use the reduced accelerations set and use the complete set otherwise)
        if index_setup == 0:
            acceleration_settings = {
                spacecraft_name: simple_accelerations_settings_spacecraft
            }
        else:
            acceleration_settings = {
                spacecraft_name: complete_accelerations_settings_spacecraft
            }

        # Create acceleration models from settings
        bodies_to_propagate = [spacecraft_name]
        central_bodies = [spacecraft_central_body]
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # Define integrator settings
        integration_step = 30.0
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            time_representation.Time(0, integration_step),
            propagation_setup.integrator.rkf_78,
        )

        ### ------------------------------------------------------------------------------------------
        ### DEFINE SET OF PARAMETERS TO BE ESTIMATED
        ### ------------------------------------------------------------------------------------------

        # For setups 0 and 1, only estimate GRAIL's initial state
        extra_parameters = []

        # For index_setup >= 2, define list of additional parameters

        # Add radiation pressure scale factors
        if index_setup >= 2:
            extra_parameters.append(
                parameters_setup.radiation_pressure_target_direction_scaling(
                    spacecraft_name, "Sun"
                )
            )
            extra_parameters.append(
                parameters_setup.radiation_pressure_target_perpendicular_direction_scaling(
                    spacecraft_name, "Sun"
                )
            )
            extra_parameters.append(
                parameters_setup.radiation_pressure_target_direction_scaling(
                    spacecraft_name, "Moon"
                )
            )
            extra_parameters.append(
                parameters_setup.radiation_pressure_target_perpendicular_direction_scaling(
                    spacecraft_name, "Moon"
                )
            )

        # Add the estimation of the manoeuvre(s) (if any are detected for the current date)
        if index_setup == 3:
            if len(relevant_manoeuvres) > 0:
                extra_parameters.append(
                    parameters_setup.quasi_impulsive_shots(spacecraft_name)
                )

        # Fit the propagated GRAIL trajectory to its reference spice trajectory. This function creates ideal position "observables" by directly
        # sampling the GRAIL spice trajectory. GRAIL's trajectory propagated with the dynamical model defined above is then fitted to these
        # spice observables, using the current set of estimated paramaters.
        estimation_output = estimation_analysis.create_best_fit_to_ephemeris(
            bodies,
            acceleration_models,
            bodies_to_propagate,
            central_bodies,
            integrator_settings,
            start_time,
            end_time,  # initial_time, final_time,
            time_representation.Time(0, 60.0),
            extra_parameters,
            results_print_frequency=3600.0 / integration_step,
        )

        # Retrieve GRAIL's post-fit estimated trajectory
        estimated_state_history = estimation_output.simulation_results_per_iteration[
            -1
        ].dynamics_results.state_history

        # Compute the difference between GRAIL's post-fit state history and its reference spice trajectory, in the RSW frame
        # (radial, along-track, cross-track).
        rsw_state_difference = get_rsw_state_difference(
            estimated_state_history,
            spacecraft_name,
            spacecraft_central_body,
            global_frame_orientation,
        )

        # Save RSW state difference w.r.t. spice trajectory
        np.savetxt(
            output_folder
            + "fit_spice_rsw_state_difference_"
            + filename_suffix
            + ".dat",
            rsw_state_difference,
            delimiter=",",
        )

        # Estimated parameters
        print("estimated parameters", estimation_output.parameter_history[:, -1])


if __name__ == "__main__":
    print("Start")
    inputs = []

    output_folder = "grail_parallel_outputs/"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Define dates for the five analyses to be run in parallel (we use the same dates as in the grail_odf_estimation.py example).
    dates = [
        datetime(2012, 4, 6),
        datetime(2012, 4, 9),
        datetime(2012, 4, 10),
        datetime(2012, 4, 11),
        datetime(2012, 4, 12),
    ]
    nb_dates = len(dates)

    # Specify the number of different setups to try when fitting GRAIL's dynamics to the reference spice trajectory.
    # In this example, "setup" refers to the combination of i) GRAIL's dynamical model and ii) the set of estimated parameters used
    # to fit the spice trajectory. Two different dynamical models are used to propagate GRAIL's dynamics:
    # model A: spherical harmonics model truncated at degree/order = 10/10 for the Moon, cannonball radiation pressure for both
    #          the Sun and the Moon, third-body perturbation from the Earth only
    # model B: spherical harmonics model up to degree/order = 256/256 for the Moon, radiation pressure: cannonball model for the Sun
    #          and complete panel model for the Moon, third-body perturbations from Earth, Sun, Venus, Mars, Jupiter, and Saturn.
    nb_setups = 4
    # The six different setups are defined as follows:
    # setup 0: model A + estimated parameters: GRAIL's initial state
    # setup 1: model B + estimated parameters: GRAIL's initial state
    # setup 2: model B + estimated parameters: GRAIL's initial state + radiation pressure scale factors
    # setup 3: model B + estimated parameters: GRAIL's initial state + radiation pressure scale factors + manoeuvres (if any)

    # Define the number of parallel runs to use for this example
    nb_parallel_runs = nb_setups * nb_dates

    # For each parallel run
    for i in range(nb_parallel_runs):

        # Retrieve the indices of the current date and current setup (dynamical model + estimated parameters) for this run
        index_setup = i // nb_dates
        index_date = i % nb_dates
        print("index_setup", index_setup)
        print("index_date", index_date)

        # First retrieve the names of all the relevant kernels and data files necessary to cover the date of interest
        (
            clock_file,
            grail_orientation_files,
            tro_files,
            ion_files,
            manoeuvres_file,
            antenna_files,
            odf_files,
            trajectory_files,
            grail_frames_def_file,
            moon_orientation_file,
            lunar_frame_file,
        ) = get_grail_files("grail_kernels/", dates[index_date], dates[index_date])

        # Construct a list of input arguments containing the arguments needed this specific parallel run.
        # These include the start and end dates, along with the names of all relevant kernels and data files that should be loaded
        inputs.append(
            [
                i,
                dates[index_date],
                index_setup,
                clock_file,
                grail_orientation_files,
                manoeuvres_file,
                trajectory_files,
                grail_frames_def_file,
                moon_orientation_file,
                lunar_frame_file,
                output_folder,
            ]
        )

    # Run parallel GRAIL fit to spice
    print("---------------------------------------------")
    print(
        "The output of each parallel fit to spice is saved in a separate file named grail_spice_fit_output_DATE_setup_x.dat, "
        "with x the index of the current setup and DATE the date of interest written as MMDDYYYYY (all output files are saved in "
        + output_folder
    )
    with mp.get_context("fork").Pool(nb_parallel_runs) as pool:
        pool.map(run_spice_fit, inputs)

    # Load and plot the results of each fit to SPICE
    for index_date in range(nb_dates):

        # Retrieve start of the current date in seconds
        start_date = time_representation.DateTime.from_python_datetime(dates[index_date]).to_epoch()

        # Retrieve string corresponding to the current date
        date_string = dates[index_date].strftime("%m/%d/%Y").replace("/", "")

        # Plot the results of the fit for the current date
        nb_subplot_cols = int(np.ceil(nb_setups / 2))
        fig, axs = plt.subplots(2, nb_subplot_cols, figsize=(10, 8))

        # Parse results for all setups
        for index_setup in range(nb_setups):

            # Retrieve post-fit difference wrt reference SPICE trajectory
            difference_rsw_wrt_spice = np.loadtxt(
                output_folder
                + "fit_spice_rsw_state_difference_"
                + date_string
                + "_setup_"
                + str(index_setup)
                + ".dat",
                delimiter=",",
            )

            # Plot the difference between the post-fit GRAIL trajectory and the reference spice kernel, in RSW frame
            row = index_setup // nb_subplot_cols
            col = index_setup % nb_subplot_cols

            axs[row, col].plot(
                (difference_rsw_wrt_spice[:, 0] - start_date) / 3600,
                difference_rsw_wrt_spice[:, 1],
                label="radial",
            )
            axs[row, col].plot(
                (difference_rsw_wrt_spice[:, 0] - start_date) / 3600,
                difference_rsw_wrt_spice[:, 2],
                label="along-track",
            )
            axs[row, col].plot(
                (difference_rsw_wrt_spice[:, 0] - start_date) / 3600,
                difference_rsw_wrt_spice[:, 3],
                label="cross-track",
            )
            axs[row, col].grid()
            axs[row, col].set_xlim([0, 24])
            axs[row, col].set_xlabel("Time [hours since start of the day]")
            axs[row, col].set_ylabel("Difference wrt spice [m]")
            axs[row, col].set_title("Setup " + str(index_setup))
            axs[row, col].legend()

        fig.suptitle("GRAIL spice fit for " + dates[index_date].strftime("%m/%d/%Y"))
        fig.tight_layout()

        plt.show()
