# IMPORTS
import numpy as np
from numpy import pi as PI
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import polyfit
from tudatpy.util import result2array
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from matplotlib import pyplot as plt
TWOPI = 2.0*PI

spice.load_standard_kernels([])

'''
########################################################################################################################
##########                                                                                                    ##########
##########                                            AUXILIARIES                                             ##########
##########                                                                                                    ##########
########################################################################################################################
'''


def get_gravitational_field(frame_name: str) -> environment_setup.gravity_field.GravityFieldSettings:

    # The gravitational field implemented here is that by Le Maistre et al. (2019).

    phobos_gravitational_parameter = 1.06e16*constants.GRAVITATIONAL_CONSTANT
    phobos_reference_radius = 14e3

    phobos_normalized_cosine_coefficients = np.array([[ 1.0,      0.0, 0.0,      0.0, 0.0],
                                                      [ 0.0,      0.0, 0.0,      0.0, 0.0],
                                                      [-0.029243, 0.0, 0.015664, 0.0, 0.0],
                                                      [ 0.0,      0.0, 0.0,      0.0, 0.0],
                                                      [ 0.0,      0.0, 0.0,      0.0, 0.0]])
    phobos_normalized_sine_coefficients = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0]])

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame = frame_name)

    return settings_to_return


def get_initial_rotational_state_at_epoch(epoch: float) -> np.ndarray:

    translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'None', epoch)
    synchronous_rotation_matrix = inertial_to_rsw_rotation_matrix(translational_state).T
    synchronous_rotation_matrix[:,:2] = -1.0*synchronous_rotation_matrix[:,:2]
    phobos_rotation_quaternion = rotation_matrix_to_quaternion_entries(synchronous_rotation_matrix)

    angular_rate = 0.000228035245  # In rad/s
    angular_velocity = np.array([0.0, 0.0, angular_rate])

    return np.concatenate((phobos_rotation_quaternion, angular_velocity))


# ANGLE INTERVAL MANAGEMENT
def bring_inside_bounds(original: float | np.ndarray, lower_bound: float,
                        upper_bound: float, include: str = 'lower') -> float | np.ndarray:

    if include not in ['upper', 'lower']:
        raise ValueError('(bring_inside_bounds): Invalid value for argument "include". Only "upper" and "lower" are '
                         'allowed. Provided: ' + include)

    scalar_types = [float, np.float32, np.float64, np.float128]
    if type(original) in scalar_types:
        return bring_inside_bounds_scalar(original, lower_bound, upper_bound, include)

    dim_num = len(original.shape)

    if dim_num == 1: to_return = bring_inside_bounds_single_dim(original, lower_bound, upper_bound, include)
    elif dim_num == 2: to_return = bring_inside_bounds_double_dim(original, lower_bound, upper_bound, include)
    else: raise ValueError('(bring_inside_bounds): Invalid input array.')

    return to_return


def bring_inside_bounds_single_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    new = np.zeros_like(original)
    for idx in range(len(new)):
        new[idx] = bring_inside_bounds_scalar(original[idx], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_double_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    lengths = original.shape
    new = np.zeros_like(original)
    for idx0 in range(lengths[0]):
        for idx1 in range(lengths[1]):
            new[idx0, idx1] = bring_inside_bounds_scalar(original[idx0, idx1], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_scalar(original: float, lower_bound: float,
                               upper_bound: float, include: str = 'lower') -> float:

    if original == upper_bound or original == lower_bound:
        if include == 'lower':
            return lower_bound
        else:
            return upper_bound

    if lower_bound < original < upper_bound:
        return original

    center = (upper_bound + lower_bound) / 2.0

    if original < lower_bound:
        reflect = True
    else:
        reflect = False

    if reflect: original = 2.0 * center - original

    dividend = original - lower_bound
    divisor = upper_bound - lower_bound
    remainder = dividend % divisor
    new = lower_bound + remainder

    if reflect: new = 2.0 * center - new

    if new == lower_bound and include == 'upper': new = upper_bound
    if new == upper_bound and include == 'lower': new = lower_bound

    return new


# FAST FOURIER TRANSFORM FUNCTIONALITIES
def get_fourier(time_history: np.ndarray, clean_signal: list = [0.0, 0]) -> tuple:

    sample_times = time_history[:,0]
    signal = time_history[:,1]

    if len(sample_times) % 2.0 != 0.0:
        sample_times = sample_times[:-1]
        signal = signal[:-1]

    if clean_signal[0] != 0.0: signal = remove_jumps(signal, clean_signal[0])
    if clean_signal[1] != 0:
        coeffs = polyfit(sample_times, signal, clean_signal[1])
        signal = signal - coeffs[0] - coeffs[1] * sample_times

    n = len(sample_times)
    dt = sample_times[1] - sample_times[0]
    frequencies = TWOPI * rfftfreq(n, dt)
    amplitudes = 2*abs(rfft(signal, norm = 'forward'))

    return frequencies, amplitudes


def remove_jumps(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

    dim_num = len(original.shape)

    if dim_num == 1: return remove_jumps_single_dim(original, jump_height, margin)
    elif dim_num == 2: return remove_jumps_double_dim(original, jump_height, margin)
    else: raise ValueError('(remove_jumps): Invalid input array.')


def remove_jumps_single_dim(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

    new = original.copy()
    u = 1.0 - margin
    l = -1.0 + margin
    for idx in range(len(new)-1):
        d = (new[idx+1] - new[idx]) / jump_height
        if d <= l: new[idx+1:] = new[idx+1:] + jump_height
        if d >= u: new[idx+1:] = new[idx+1:] - jump_height

    return new


def remove_jumps_double_dim(original: np.array, jump_height: float, margin: float = 0.03) -> np.ndarray:

    new = original.copy()
    u = 1.0 - margin
    l = -1.0 + margin
    for col in range(new.shape[1]):
        for row in range(new.shape[0]-1):
            d = ( new[row+1,col] - new[row,col] ) / jump_height
            if d <= l: new[row+1:,col] = new[row+1:,col] + jump_height
            if d >= u: new[row+1:,col] = new[row+1:,col] - jump_height

    return new


# NORMAL MODE COMPUTATION
def get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor: np.ndarray, mean_motion: float) -> float:

    # From Rambaux (2012) "Rotational motion of Phobos".

    A = inertia_tensor[0,0]
    B = inertia_tensor[1,1]
    C = inertia_tensor[2,2]
    gamma = (B - A) / C

    return mean_motion * np.sqrt(3*gamma)


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          ENVIRONMENT                                               ##########
##########                                                                                                    ##########
########################################################################################################################
'''


# WE CREATE EVERYTHING THAT IS NOT PHOBOS
bodies_to_create = ["Sun", "Earth", "Mars", "Deimos", "Jupiter"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"  # WATCH OUT! This represents the Earth's equatorial reference frame at the J2000 epoch.
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# AND NOW WE CREATE PHOBOS
body_settings.add_empty_settings('Phobos')
body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed')
body_settings.get('Phobos').gravity_field_settings.scaled_mean_moment_of_inertia = 0.43

# AND NOW WE CREATE THE BODIES OBJECT
bodies = environment_setup.create_system_of_bodies(body_settings)

'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          COMMON SETTINGS                                           ##########
##########                                                                                                    ##########
########################################################################################################################
'''

# INTEGRATOR SETTINGS
# Here, we will select an RKDP7(8) integrator working in a fixed-step regime with a step size of 5 minutes.
time_step = 300.0  # This is 5 minutes in seconds.
coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                  coefficients,
                                                                                  time_step,
                                                                                  time_step,
                                                                                  np.inf, np.inf)

# INITIAL TIME
initial_epoch = 0.0  # This is the J2000 epoch.


# TERMINATION CONDITION
# We will run a simulation of 30 days.
simulation_time = 30.0*constants.JULIAN_DAY
termination_condition = propagation_setup.propagator.time_termination(initial_epoch + simulation_time,
                                                                      terminate_exactly_on_final_condition = True)


# DEPENDENT VARIABLES
dependent_variables = [ propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),
                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),
                        propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos')
                      ]


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          TRANSLATION                                               ##########
##########                                                                                                    ##########
########################################################################################################################
'''


# CENTRAL BODIES AND BODIES TO PROPAGATE
central_bodies =['Mars']
bodies_to_propagate = ['Phobos']


# ACCELERATION MODEL
third_body_force = propagation_setup.acceleration.point_mass_gravity()
acceleration_settings_on_phobos = dict( Mars    = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 4, 4)],
                                        Sun     = [propagation_setup.acceleration.point_mass_gravity()],
                                        Earth   = [propagation_setup.acceleration.point_mass_gravity()],
                                        Deimos  = [propagation_setup.acceleration.point_mass_gravity()],
                                        Jupiter = [propagation_setup.acceleration.point_mass_gravity()] )

acceleration_settings = {'Phobos': acceleration_settings_on_phobos}
acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate,
                                                                  central_bodies)


# INITIAL STATE
initial_translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'NONE', initial_epoch)


# PROPAGATION SETTINGS
translational_propagator_settings = propagation_setup.propagator.translational( central_bodies,
                                                                                acceleration_model,
                                                                                bodies_to_propagate,
                                                                                initial_translational_state,
                                                                                initial_epoch,
                                                                                integrator_settings,
                                                                                termination_condition )


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          ROTATION                                                  ##########
##########                                                                                                    ##########
########################################################################################################################
'''


# BODIES TO PROPAGATE
bodies_to_propagate = ['Phobos']


# TORQUE SETTINGS
torque_settings_on_phobos = dict( Mars    = [propagation_setup.torque.spherical_harmonic_gravitational(4,4)],
                                  Sun     = [propagation_setup.torque.spherical_harmonic_gravitational(4,4)],
                                  Earth   = [propagation_setup.torque.spherical_harmonic_gravitational(4,4)],
                                  Deimos  = [propagation_setup.torque.spherical_harmonic_gravitational(4,4)],
                                  Jupiter = [propagation_setup.torque.spherical_harmonic_gravitational(4,4)] )

torque_settings = {'Phobos': torque_settings_on_phobos}
torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

# INITIAL STATE
initial_rotational_state = get_initial_rotational_state_at_epoch(initial_epoch)


# PROPAGATION SETTINGS
rotational_propagator_settings = propagation_setup.propagator.rotational( torque_model,
                                                                          bodies_to_propagate,
                                                                          initial_rotational_state,
                                                                          initial_epoch,
                                                                          integrator_settings,
                                                                          termination_condition )


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                  COMBINATION AND SIMULATION                                        ##########
##########                                                                                                    ##########
########################################################################################################################
'''

# MULTI-TYPE PROPAGATOR
propagator_list = [translational_propagator_settings, rotational_propagator_settings]
combined_propagator_settings = propagation_setup.propagator.multitype( propagator_list,
                                                                       integrator_settings,
                                                                       initial_epoch,
                                                                       termination_condition,
                                                                       output_variables = dependent_variables)

# DYNAMICS SIMULATION
simulator = numerical_simulation.create_dynamics_simulator(bodies, combined_propagator_settings)
state_history = simulator.state_history
dependent_variable_history = simulator.dependent_variable_history


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          POST-PROCESS                                              ##########
##########                                                                                                    ##########
########################################################################################################################
'''


## PLOTS

# INDEX:      1,   2,   3,   4,     5,    6,     7,  8,   9,   10,     11,        12,        13
# STATE:      x,   y,   z,   vx,    vy,   vz,    q0, q1,  q2,  q3,     ang_vel_1, ang_vel_2, ang_vel_3
# DEPENDENTS: a,   e,   i,   omega, RAAN, theta, r,  lat, lon, euler3, euler1,    euler3

states_array = result2array(state_history)
dependents_array = result2array(dependent_variable_history)
epochs = states_array[:,0] / constants.JULIAN_DAY
time_label = 'Time since J2000 [days]'


plt.figure()
plt.plot(epochs, dependents_array[:,1] / 1e3)
plt.grid()
plt.xlabel(time_label)
plt.ylabel(r'$a$ [km]')
plt.title('Semimajor axis')

plt.figure()
plt.plot(epochs, dependents_array[:,2])
plt.grid()
plt.xlabel(time_label)
plt.ylabel(r'$e$ [-]')
plt.title('Eccentricity')

plt.figure()
plt.plot(epochs, np.degrees(dependents_array[:,3]), label = r'$i$')
plt.plot(epochs, np.degrees(dependents_array[:,4]), label = r'$\omega$')
plt.plot(epochs, np.degrees(dependents_array[:,5]), label = r'$\Omega$')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Angle [º]')
plt.title('Orbit\'s Euler angles')

plt.figure()
plt.plot(epochs, np.degrees(dependents_array[:,9]), label = r'Lon')
plt.plot(epochs, np.degrees(dependents_array[:,8]), label = r'Lat')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Coordinate [º]')
plt.title('Mars\' coordinates in Phobos\' sky')

cmap = plt.get_cmap('PRGn')
fig, axis = plt.subplots()
axis.scatter(np.degrees(dependents_array[:,9]), np.degrees(dependents_array[:,8]), c = epochs, cmap = cmap)
axis.grid()
axis.set_xlabel('Longitude [º]')
axis.set_ylabel('Latitude [º]')
axis.set_title('Mars\' coordinates in Phobos\' sky')
fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs[0], epochs[-1])), ax=axis,
                 orientation='vertical', label=time_label)

plt.figure()
plt.plot(epochs, np.degrees(bring_inside_bounds(dependents_array[:,10], -PI, PI, include = 'upper')), label = r'$\Psi$')
plt.plot(epochs, np.degrees(dependents_array[:,11]), label = r'$\theta$')
# plt.plot(epochs, np.degrees(dependents_array[:,12]), label = r'$\varphi$')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Angle [º]')
plt.title('Phobos\' Euler angles')

mean_motion = 0.0002278563609852602
normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, mean_motion)
librations = bring_inside_bounds(dependents_array[:,8:10], -PI, PI, 'upper')
lon_lib_freq, lon_lib_amp = get_fourier(np.hstack((np.atleast_2d(dependents_array[:,0]).T, np.atleast_2d(librations[:,1]).T)), [TWOPI, 1])
lat_lib_freq, lat_lib_amp = get_fourier(np.hstack((np.atleast_2d(dependents_array[:,0]).T, np.atleast_2d(librations[:,0]).T)), [TWOPI, 1])
plt.figure()
plt.loglog(lon_lib_freq * 86400.0, np.degrees(lon_lib_amp), marker='.', label='Lon')
# plt.loglog(lat_lib_freq * 86400.0, np.degrees(lat_lib_amp), marker='.', label='Lat')
plt.gca().set_ylim(bottom=1e-8)
plt.axvline(mean_motion * 86400.0, ls='dashed', c='r', linewidth = 1.0, label='Phobos\' mean motion (and integer multiples)')
plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', linewidth = 1.0, label='Longitudinal normal mode')
plt.axvline(2 * mean_motion * 86400.0, ls='dashed', c='r', linewidth = 1.0)
plt.axvline(3 * mean_motion * 86400.0, ls='dashed', c='r', linewidth = 1.0)
plt.title('Longitudinal libration frequency content')
plt.xlabel(r'$\omega$ [rad/day]')
plt.ylabel(r'$A [º]$')
plt.grid()
plt.legend()

plt.show()


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                           DAMPING                                                  ##########
##########                                                                                                    ##########
########################################################################################################################
'''


phobos_mean_rotational_rate = 0.000228035245  # In rad/s
# As dissipation times, we will start with 4h and keep duplicating the damping time in each iteration. In the final
# iteration, a damping time of 4096h means a propagation time of 40960h, which is a bit over 4.5 years.
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0])*3600.0)
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         combined_propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)
damped_state_history = damping_results.forward_backward_states[-1][1]
damped_dependent_variable_history = damping_results.forward_backward_dependent_variables[-1][1]


'''
########################################################################################################################
##########                                                                                                    ##########
##########                                          POST-PROCESS                                              ##########
##########                                                                                                    ##########
########################################################################################################################
'''


## PLOTS

# INDEX:      1,   2,   3,   4,     5,    6,     7,  8,   9,   10,     11,        12,        13
# STATE:      x,   y,   z,   vx,    vy,   vz,    q0, q1,  q2,  q3,     ang_vel_1, ang_vel_2, ang_vel_3
# DEPENDENTS: a,   e,   i,   omega, RAAN, theta, r,  lat, lon, euler3, euler1,    euler3

epochs_of_first_30_days = [epoch for epoch in list(state_history.keys()) if epoch <= 30.0*constants.JULIAN_DAY]
reduced_history = dict.fromkeys(epochs_of_first_30_days)
for epoch in epochs_of_first_30_days:
    reduced_history[epoch] = damped_state_history[epoch]
damped_states_array = result2array(reduced_history)
for epoch in epochs_of_first_30_days:
    reduced_history[epoch] = damped_dependent_variable_history[epoch]
damped_dependents_array = result2array(reduced_history)
epochs = damped_states_array[:,0] / constants.JULIAN_DAY
time_label = 'Time since J2000 [days]'

plt.figure()
plt.plot(epochs, damped_dependents_array[:,1] / 1e3)
plt.grid()
plt.xlabel(time_label)
plt.ylabel(r'$a$ [km]')
plt.title('Semimajor axis')

plt.figure()
plt.plot(epochs, damped_dependents_array[:,2])
plt.grid()
plt.xlabel(time_label)
plt.ylabel(r'$e$ [-]')
plt.title('Eccentricity')

plt.figure()
plt.plot(epochs, np.degrees(damped_dependents_array[:,3]), label = r'$i$')
plt.plot(epochs, np.degrees(damped_dependents_array[:,4]), label = r'$\omega$')
plt.plot(epochs, np.degrees(damped_dependents_array[:,5]), label = r'$\Omega$')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Angle [º]')
plt.title('Orbit\'s Euler angles')

plt.figure()
plt.plot(epochs, np.degrees(damped_dependents_array[:,9]), label = r'Lon')
plt.plot(epochs, np.degrees(damped_dependents_array[:,8]), label = r'Lat')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Coordinate [º]')
plt.title('Mars\' coordinates in Phobos\' sky')

cmap = plt.get_cmap('PRGn')
fig, axis = plt.subplots()
axis.scatter(np.degrees(damped_dependents_array[:,9]), np.degrees(damped_dependents_array[:,8]), c = epochs, cmap = cmap)
axis.grid()
axis.set_xlabel('Longitude [º]')
axis.set_ylabel('Latitude [º]')
axis.set_title('Mars\' coordinates in Phobos\' sky')
fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs[0], epochs[-1])), ax=axis,
             orientation='vertical', label=time_label)

plt.figure()
plt.plot(epochs, np.degrees(bring_inside_bounds(damped_dependents_array[:,10], -PI, PI, include = 'upper')), label = r'$\Psi$')
plt.plot(epochs, np.degrees(damped_dependents_array[:,11]), label = r'$\theta$')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,12]), label = r'$\varphi$')
plt.grid()
plt.legend()
plt.xlabel(time_label)
plt.ylabel(r'Angle [º]')
plt.title('Phobos\' Euler angles')

mean_motion = 0.0002278563609852602
normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, mean_motion)
damped_librations = bring_inside_bounds(damped_dependents_array[:,8:10], -PI, PI, 'upper')
damped_lon_lib_freq, damped_lon_lib_amp = get_fourier(np.hstack((np.atleast_2d(damped_dependents_array[:,0]).T, np.atleast_2d(damped_librations[:,1]).T)), [TWOPI, 1])
damped_lat_lib_freq, damped_lat_lib_amp = get_fourier(np.hstack((np.atleast_2d(damped_dependents_array[:,0]).T, np.atleast_2d(damped_librations[:,0]).T)), [TWOPI, 1])
plt.figure()
plt.loglog(damped_lon_lib_freq * 86400.0, np.degrees(damped_lon_lib_amp), marker='.', label='Lon')
# plt.loglog(lat_lib_freq * 86400.0, np.degrees(lat_lib_amp), marker='.', label='Lat')
plt.gca().set_ylim(bottom=1e-8)
plt.axvline(mean_motion * 86400.0, ls='dashed', c='r', linewidth = 1.0, label='Phobos\' mean motion (and integer multiples)')
plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', linewidth = 1.0, label='Longitudinal normal mode')
plt.axvline(2 * mean_motion * 86400.0, ls='dashed', linewidth = 1.0, c='r')
plt.axvline(3 * mean_motion * 86400.0, ls='dashed', linewidth = 1.0, c='r')
plt.title('Longitudinal libration frequency content')
plt.xlabel(r'$\omega$ [rad/day]')
plt.ylabel(r'$A [º]$')
plt.grid()
plt.legend()

plt.show()