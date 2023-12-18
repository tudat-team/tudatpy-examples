
'''
Copyright (c) 2010-2023, Delft University of Technology
All rigths reserved
This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

## IMPORTANT
"""
This example requires the `tudatpy.trajectory_design.porkchop` module.
Please ensure that your version of tudatpy does include this module.
"""

## Context
"""
This example shows how the tudatpy porkchop module can be used to choose
an optimal launch and arrival date for an Earth-Mars transfer. By default,
the porkchop module uses a Lambert arc to compute the ΔV required to
depart from the departure body (Earth in this case) and be captured by the 
target body (in this case Mars).
Users can provide a custom function to calculate the ΔV required for any
given transfer. This can be done by supplying a `callable` (a function)
to the `porkchop` function via the argument
    function_to_calculate_delta_v
This opens the possibility to calculate the ΔV required for any transfer
accounting for course correction manoeuvres along a planned trajectory.
"""

## How this example is organized
"""
This example consists of 3 sections:
1. The imports, where we import the required modules
2. Data management, where we define the file where the porkchop data will be saved
3. The porkchop itself, which will only be recalculated if the user requests it
These sections are marked with Matlab-style #%% comments. If you use
the VSCode Jupyter extension this will allow you to run each section
in turn and fold the sections to help you navigate the contents of
the file.
"""

#--------------------------------------------------------------------
#%% IMPORT STATEMENTS
#--------------------------------------------------------------------

# General imports
import os
import pickle

# Tudat imports
from tudatpy import constants
from tudatpy.trajectory_design.porkchop import porkchop, plot_porkchop

#--------------------------------------------------------------------
#%% DATA MANAGEMENT
#--------------------------------------------------------------------

# File
data_file = 'porkchop.pkl'

# Whether to recalculate the porkchop plot or use saved data
RECALCULATE_delta_v = input(
    '\n    Recalculate ΔV for porkchop plot? [y/N] '
).strip().lower() == 'y'
print()

#--------------------------------------------------------------------
#%% PORKCHOP
#--------------------------------------------------------------------

# Define departure and target bodies
departure_body = 'Earth'
target_body = 'Mars'

if not os.path.isfile(data_file) or RECALCULATE_delta_v:

    # Tudat imports
    from tudatpy.kernel.interface import spice_interface
    from tudatpy.kernel.astro.time_conversion import DateTime
    from tudatpy.kernel.numerical_simulation import environment_setup

    #--------------------------------------------------------------------
    #%% PORKCHOP PARAMETERS
    #--------------------------------------------------------------------

    # Load spice kernels
    spice_interface.load_standard_kernels( )

    # Define global frame orientation
    global_frame_orientation = 'ECLIPJ2000'

    # Create bodies
    bodies_to_create = ['Sun', 'Venus', 'Earth', 'Moon', 'Mars', 'Jupiter', 'Saturn']
    global_frame_origin = 'Sun'
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

    # Create environment model
    bodies = environment_setup.create_system_of_bodies(body_settings)

    earliest_departure_time = DateTime(2005,  4,  30)
    latest_departure_time   = DateTime(2005, 10,   7)

    earliest_arrival_time   = DateTime(2005, 11,  16)
    latest_arrival_time     = DateTime(2006, 12,  21)

    # Set time resolution IN DAYS as 0.5% of the smallest window (be it departure, or arrival)
    # This will result in fairly good time resolution, at a runtime of approximately 10 seconds
    # Tune the time resolution to obtain results to your liking!
    time_window_percentage = 0.5
    time_resolution = time_resolution = min(
            latest_departure_time.epoch() - earliest_departure_time.epoch(),
            latest_arrival_time.epoch()   - earliest_arrival_time.epoch()
    ) / constants.JULIAN_DAY * time_window_percentage / 100

    [departure_epochs, arrival_epochs, ΔV] = porkchop(
        bodies,
        global_frame_orientation,
        departure_body,
        target_body,
        earliest_departure_time,
        latest_departure_time,
        earliest_arrival_time,
        latest_arrival_time,
        time_resolution
    )
    pickle.dump(
        [departure_epochs, arrival_epochs, ΔV],
        open(data_file, 'wb')
    )

else:

    [departure_epochs, arrival_epochs, ΔV] = pickle.load(
        open(data_file, 'rb')
    )

    # Cases
    cases = [
        {'C3': False, 'total': False, 'threshold': 15 , 'filename': 'figures/ΔV.png'},
        {'C3': False, 'total': True,  'threshold': 15 , 'filename': 'figures/Δ_tot.png'},
        {'C3': True,  'total': False, 'threshold': 42 , 'filename': 'figures/C3.png'},
        {'C3': True,  'total': True,  'threshold': 100, 'filename': 'figures/C3_tot.png'}
    ]

    for case in cases:
        plot_porkchop(
            departure_body   = departure_body,
            target_body      = target_body,
            departure_epochs = departure_epochs,
            arrival_epochs   = arrival_epochs,
            delta_v          = ΔV,
            save             = True,
            **case
        )