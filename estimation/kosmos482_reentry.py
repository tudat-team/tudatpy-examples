######################################################################################################
#                                                                                                    #
#           Tudat script to propagate an object to reentry (= chosen end altitude)                   #
#                                                                                                    #
#                                                                                                    #
#                             Marco Langbroek, Dominic Dirkx                                         #
#                             Delft University of Technology                                         #
#                            faculty of Aerospace Engineering                                        #
#                                                                                                    #
#                                                                                                    #
######################################################################################################

# ****************************************************************************************************
#                                     IMPORTANT NOTE:                                                #
#                                                                                                    #
#                   REPLACE THE SPACEWEATHER FILE sw19571001.txt WITH                                #
#                  AN UP TO DATE VERSION BEFORE RUNNING THIS SCRIPT!!!                               #
#                                                                                                    #
# ****************************************************************************************************

# determine program start (to later calculate runtime duration)
from datetime import datetime
start_time_program = datetime.now()
startstring = str(start_time_program)
print('')
print('run has started at: '+ startstring + '\n')
print('be patient....\n')

import math
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import geopandas as gpd
import numpy as np
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, environment, propagation_setup, propagation
from tudatpy.astro import time_representation, time_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Load spice kernels
spice.load_standard_kernels()

# Create time scale converter object
time_scale_converter = time_conversion.default_time_scale_converter( )

#####---------------------------------------------------------------------------------------#####

# ******************  BASIC OBJECT AND SIMULATION INPUT DATA, OPTIONS  ************************ #

# Object ID
objname = "KOSMOS 482 DESCENT CRAFT"
catnr = "6073"
cospar = "1972-023E"

#---------------------------------------------------------------------------------------------------------------#
# TLE
lineone = "1  6073U 72023E   25130.02495443  .08088373  12542-4  65849-4 0  9993"  # TLE line 1
linetwo = "2  6073  51.9455 241.9030 0035390 103.6551  63.6433 16.48653575751319"  # TLE line 2
#---------------------------------------------------------------------------------------------------------------#

# object mass
gewicht = 480 # kg

# DRAG AND SRP AREA, DRAG COEFFICIENT
reference_area_drag = 0.7854 # Average projection area of the spacecraft in m^2
reference_area_radiation = 0.7854  # Average projection area of object for SRP. keep 0.0 to ignore SRP
drag_coefficient = 2.2 # drag coefficient

# CUT-OFF ALTITUDE FOR SIMULATION
altlimit = 50.0e3  #meters  (standard 50 km = 50.0e3)

# SET SIMULATION START EPOCH
# THIS SHOULD EQUAL THE TIME OF THE TLE EPOCH - NOTE: no leading zeros, format yyyy, m, d, h, m, s.sss
simulation_start_epoch_utc = DateTime(2025, 5, 10, 0, 35, 56.06).epoch()

# SET  SIMULATION END EPOCH (cuts off run if altitude criterion not met before)
simulation_end_epoch_utc = DateTime(2026, 2, 1, 0, 0, 0).epoch()

# SET OPTION FULL OUTPUT OR ONLY END RESULT? ('full', 'selected',"J2K",'endonly')
     
outopt = 'selected'

    # full = selected data, J2K States and end result out
    # selected = only selected data and end result out
    # J2K = only State and end result out
    # endonly = only end result out

# SET OPTIONAL OUTPUT FILE NAME SUFFIX
filenamesuf = "_KMOS482_480kg_FINAL"

# SET OPTION FOR SAVE INTERVAL IF DESIRED TO NOT SAVE EACH STEP (yes/no?)
interval_set = 'no'
interval = 20  #seconds   10 days = 864000 seconds 1 day = 86400 seconds 1 hr = 3600 seconds

#####----------------------------------------------------------------------------------------#####

# *********************************  START BASIC SCRIPT  *************************************** #

# start and end epoch of simulation conversion from UTC to tdb
simulation_start_epoch_tdb = time_scale_converter.convert_time(
  input_scale = time_conversion.utc_scale,
  output_scale = time_conversion.tdb_scale,
  input_value = simulation_start_epoch_utc )
simulation_end_epoch_tdb = time_scale_converter.convert_time(
  input_scale = time_conversion.utc_scale,
  output_scale = time_conversion.tdb_scale,
  input_value = simulation_end_epoch_utc )

# some values to strings for output later
massstring = str(gewicht)
areastring = str(reference_area_drag)
altlimitstring = str(altlimit)

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create Earth rotation model
body_settings.get("Earth").rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
                environment_setup.rotation_model.iau_2006,
                global_frame_orientation )
body_settings.get("Earth").gravity_field_settings.associated_reference_frame = "ITRS"

# create atmosphere settings and add to body settings of body "Earth"
body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

# Create earth shape model
body_settings.get("Earth").shape_settings = environment_setup.shape.oblate_spherical( 6378137.0, 1.0 / 298.257223563)

# Create empty body settings for the satellite, generic name 'candidate'
body_settings.add_empty_settings("candidate")

# Create aerodynamic coefficient interface settings
# reference_area_drag already defined earlier!
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)

# Add the aerodynamic interface to the body settings
body_settings.get("candidate").aerodynamic_coefficient_settings = aero_coefficient_settings

# Create radiation pressure settings
# reference_area_radiation already defined earlier!
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("candidate").radiation_pressure_target_settings = vehicle_target_settings

# Add body mass
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("candidate").mass = gewicht  #mass in kg, already set earlier!

# Define bodies that are propagated
bodies_to_propagate = ["candidate"]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on reentry candidate by Sun and Earth.
accelerations_settings_candidate = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {"candidate": accelerations_settings_candidate}

# Create the acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Retrieve the initial state of the reentry candidate using Two-Line-Elements (TLEs)
candidate_tle = environment.Tle(
  lineone, linetwo
)
candidate_ephemeris = environment.TleEphemeris( "Earth", "J2000", candidate_tle, False )
initial_state = candidate_ephemeris.cartesian_state( simulation_start_epoch_tdb )

# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.altitude("candidate", "Earth"),
    propagation_setup.dependent_variable.geodetic_latitude("candidate", "Earth"),
    propagation_setup.dependent_variable.longitude("candidate", "Earth"),
    propagation_setup.dependent_variable.periapsis_altitude("candidate", "Earth"),
    propagation_setup.dependent_variable.apoapsis_altitude("candidate", "Earth"),
    propagation_setup.dependent_variable.body_fixed_groundspeed_velocity("candidate", "Earth")
]

# Define a termination condition to stop once altitude goes below a certain value (defined earlier!)
termination_altitude_settings = propagation_setup.propagator.dependent_variable_termination(
    dependent_variable_settings=propagation_setup.dependent_variable.altitude("candidate", "Earth"),
    limit_value=altlimit,
    use_as_lower_limit=True)
# Define a termination condition to stop after a given time (to avoid an endless skipping re-entry)
termination_time_settings = propagation_setup.propagator.time_termination(simulation_end_epoch_tdb)
# Combine the termination settings to stop when one of them is fulfilled
combined_termination_settings = propagation_setup.propagator.hybrid_termination(
    [termination_altitude_settings, termination_time_settings], fulfill_single_condition=True )

# Create numerical integrator settings
# Create RK settings / RK7(8)
control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance( 1.0E-10, 1.0E-10 )
validation_settings = propagation_setup.integrator.step_size_validation( 0.001, 2700.0 )
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
    initial_time_step = 60.0,
    coefficient_set = propagation_setup.integrator.rkf_78,
    step_size_control_settings = control_settings,
    step_size_validation_settings = validation_settings )

# Create the propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch_tdb,
    integrator_settings,
    combined_termination_settings,
    output_variables=dependent_variables_to_save
)

# Set output data save interval if defined
if interval_set == 'yes':
    propagator_settings.processing_settings.results_save_frequency_in_seconds = interval
    propagator_settings.processing_settings.results_save_frequency_in_steps = 0

# Create the simulation objects and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

# Extract the resulting simulation dependent variables
dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
# Convert the dependent variables from a dictionary to a numpy array
dependent_variables_array = result2array(dependent_variables)

# determine and print runtime duration
end_time_program = datetime.now()
print('FINISHED. Runtime duration: {}'.format(end_time_program - start_time_program))


# WRITE RESULTS TO SCREEN AND TO TEXT FILES

data = dependent_variables_array
length = len(data)
lastline= (data[length-1])

print(' ')
print('mass: ' + massstring + ' kg')
print('drag area: ' + areastring +' m^2')
print('altitude limit: ' + altlimitstring +' meter')
print(' ')

# parse reentry date and position in human-readable format
# time and position
eindtijd = lastline[0]
altid = (lastline[1])/1000.0
lat = math.degrees(lastline[2])
lon = math.degrees(lastline[3])
latstring =  "{:.2f}".format(lat)
lonstring = "{:.2f}".format(lon)
altstring = "{:.3f}".format(altid)
# integration window duration to reentry
duur = eindtijd - simulation_start_epoch_tdb
uren = (duur/3600.0)
urenstring = "{:.3f}".format(uren)
dagen = uren/24.0
dagenstring = "{:.3f}".format(dagen)

# reentry time
import datetime
date_1 = datetime.datetime(2000,1,1,12,0,0)
eindtijd = eindtijd - 64.184  # tdb to UTC
eindtijd_uren = (eindtijd/3600.0)
eindtijd_dagen = eindtijd_uren/24.0
end_date = date_1 + datetime.timedelta(days=eindtijd_dagen)
reentrydatestring = str(end_date)
propstart = date_1 + datetime.timedelta(days=((simulation_start_epoch_utc/3600.0)/24.0))
propstartstring = str(propstart)
propstartstring = propstartstring + " UTC"
propendstring = str(end_date)
propendstring = propendstring + " UTC"

# get uncertainty estimate (25% of integration window duration)
sigm = 0.25 * uren    # sigma defined as 25% of time between TLE epoch and reentry
if sigm < 26.0:
    formatted_number = "%.2f" % sigm
    sigm_string = str(formatted_number)
    sigm_string = sigm_string + ' hr'
if sigm < 1.0:
    sigm_mins = 0.25 *(duur/60.0)
    formatted_number = "%.2f" % sigm_mins
    sigm_string = str(formatted_number)
    sigm_string = sigm_string + ' min'
if sigm >= 26.0:
    sigm_days = 0.25 * dagen
    formatted_number = "%.2f" % sigm_days
    sigm_string = str(formatted_number)
    sigm_string = sigm_string + ' days'

# print data to screen
print(' ')
print('propagation start:  ' + propstartstring)
print('propagation end:    ' + propendstring)
print(" ")

if (altid * 1e3) > altlimit:
    print("OBJECT DID NOT REENTER WITHIN DEFINED TIMESPAN...")
else:
    print('final altitude ' + altstring + ' km')
    print(" ")
    print('reentry after ' + urenstring + ' hours  = ' + dagenstring + ' days')
    print(" ")
    print ('REENTRY AT:')
    print(reentrydatestring + ' UTC  +-  ' + sigm_string)
    print('lat: ' + latstring + '   lon: ' + lonstring)
print(" ")
print(" ")

# SAVE THE OUTPUT TO TEXT FILE(S)

from numpy import savetxt

# Save intermediate data to a comma-delimited txt file if option was chosen earlier
if outopt == 'full':
    savetxt('variables_out' + filenamesuf + '.txt', dependent_variables_array, delimiter=',')
    savetxt('J2Kstate_out' + filenamesuf + '.txt', states_array, delimiter=',')

if outopt == 'selected':
    savetxt('variables_out' + filenamesuf + '.txt', dependent_variables_array, delimiter=',')

if outopt == 'J2K':
    savetxt('J2Kstate_out' + filenamesuf + '.txt', states_array, delimiter=',')
    
# Save final data (reentry date and position) to a text file
savetxt('reentrytime_out' + filenamesuf + '.txt', lastline, delimiter=',')

# Re-open the file in append mode and append
file = open('reentrytime_out' + filenamesuf + '.txt', 'a')

#Append reentry info to the file
file.write('\n')
file.write('Runtime duration: {}'.format(end_time_program - start_time_program))
file.write('\n')
file.write('\n' + 'OBJECT: ' + objname + '\n')
file.write('CATNR:  ' + catnr + '\n')
file.write('COSPAR: ' + cospar + '\n')
file.write('\n')
file.write(lineone + '\n')
file.write(linetwo + '\n')
file.write('\n')
file.write('mass: ' + massstring + ' kg\n')
file.write('drag area: ' + areastring +' m^2\n')
file.write('altitude limit: ' + altlimitstring +' meter\n' + '\n')
file.write('propagation start: ' + propstartstring + '\n')
file.write('propagation end:   ' + propendstring + '\n')
file.write('final altitude:    ' + altstring + '\n')
file.write('\n')
if (altid * 1e3) > altlimit:
    file.write('OBJECT DID NOT REENTER IN THIS TIMESPAN\n')
else:
    file.write('reentry after ' + dagenstring + ' days\n')
    file.write('\n')
    file.write('REENTRY AT:\n')
    file.write( reentrydatestring + ' UTC  +-  ' + sigm_string+'\n')
    file.write('lat: ' + latstring + '\n')
    file.write('lon: ' + lonstring + '\n')

# Close the file
file.close()
print("output data have been written to files in home directory")
print(" ")

# Load cities data
cities = gpd.read_file("ne_10m_populated_places.shp")
big_cities = cities[cities['POP_MAX'] > 1e6]

# Ground track data
latitudes = np.degrees(dependent_variables_array[:, 2])
longitudes = (np.degrees(dependent_variables_array[:, 3]) + 180) % 360 - 180
times = dependent_variables_array[:, 0]

times_since = (np.array(times) - times[0])/3600

utc_times = [time_representation.DateTime.to_python_datetime(time_representation.DateTime.from_epoch(time)) for time in times]

# Line segments for colored path
points = np.array([longitudes, latitudes]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(times_since.min(), times_since.max())
lc = LineCollection(segments, cmap='Purples', norm=norm, linewidth=2, transform=ccrs.Geodetic())
lc.set_array(times_since)

# Satellite imagery tiler
tiler = cimgt.QuadtreeTiles()  # Good for testing
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=tiler.crs)
#ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
ax.add_image(tiler, 4)

# Add colored ground track
ax.add_collection(lc)
cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', pad=0.03)
cbar.set_label(f'Elapsed Hours Since {utc_times[0]}', fontsize = 13)

# Start/End markers
ax.plot(longitudes[0], latitudes[0], 'yo', markersize=6, transform=ccrs.PlateCarree(), label='Start')
ax.plot(longitudes[-1], latitudes[-1], 'ro', markersize=6, transform=ccrs.PlateCarree(), label=f"Reentry ({reentrydatestring[:16]} UTC +- {sigm_string})")

# Add cities > 1M population
ax.scatter(
    big_cities.geometry.x,
    big_cities.geometry.y,
    color='red',
    s=2,
    transform=ccrs.PlateCarree(),
    label='Cities > 1M'
)

# Final touches
plt.legend(loc='lower left', fontsize='small')
plt.show()