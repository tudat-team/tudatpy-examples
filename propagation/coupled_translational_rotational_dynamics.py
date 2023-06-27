# IMPORTS
from example_auxiliaries import *

from tudatpy.util import result2array
from tudatpy.kernel.numerical_simulation import propagation_setup


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                CREATE UNIVERSE

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
field_sources = ['Le Maistre', 'Scheeres']
field_types = ['FULL', 'QUAD', '1x1', '2x2', '3x3']


spice.load_standard_kernels()

# WE CREATE EVERYTHING THAT IS NOT PHOBOS
bodies_to_create = ["Sun", "Earth", "Mars", "Deimos", "Jupiter"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"  # WATCH OUT! This represents the Earth's equatorial reference frame at the J2000 epoch.
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# AND NOW WE CREATE PHOBOS
body_settings.add_empty_settings('Phobos')
body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')

body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed', field_type, field_source)
body_settings.get('Phobos').gravity_field_settings.scaled_mean_moment_of_inertia = 0.420798  # I = (I_xx + I_yy + I_zz)/3. Values from Le Maistre.

# AND NOW WE CREATE THE BODIES OBJECT
bodies = environment_setup.create_system_of_bodies(body_settings)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                COMMON SETTINGS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# INTEGRATOR SETTINGS
# Here, we will select an RKDP7(8) integrator working in a fixed-step regime with a step size of 5 minutes. (For example, why not.)
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
termination_condition = propagation_setup.propagator.time_termination(initial_epoch + simulation_time, terminate_exactly_on_final_condition = True)


# DEPENDENT VARIABLES
dependent_variables_to_save = [ propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),
                                propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos')
                              ]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                TRANSLATIONAL DYNAMICS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)


# INITIAL STATE
initial_translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'NONE', initial_epoch)  # We will just pick the closest to a default.


# PROPAGATION SETTINGS
translational_propagator_settings = propagation_setup.propagator.translational( central_bodies,
                                                                                acceleration_model,
                                                                                bodies_to_propagate,
                                                                                initial_translational_state,
                                                                                initial_epoch,
                                                                                integrator_settings,
                                                                                termination_condition )

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                ROTATIONAL DYNAMICS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                COMBINED DYNAMICS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# MULTI-TYPE PROPAGATOR
propagator_list = [translational_propagator_settings, rotational_propagator_settings]
combined_propagator_settings = propagation_setup.propagator.multitype( propagator_list,
                                                                       integrator_settings,
                                                                       initial_epoch,
                                                                       termination_condition,
                                                                       output_variables = dependent_variables_to_save)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                SIMULATE DYNAMICS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# DYNAMICS SIMULATION
simulator = numerical_simulation.create_dynamics_simulator(bodies, combined_propagator_settings)
state_history = simulator.state_history
dependent_variable_history = simulator.dependent_variable_history

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                PLOTS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# KEPLER ELEMENTS
# PHOBOS-CENTERED POSITION OF MARS IN SPHERICAL COORDINATES
# EULER ANGLES OF PHOBOS WRT INERTIAL SPACE, BOTH IN TIME AND FREQUENCY DOMAIN

### INDEX:      1,   2,   3,   4,     5,    6,     7,  8,   9,   10,     11,        12,        13
### STATE:      x,   y,   z,   vx,    vy,   vz,    q0, q1,  q2,  q3,     ang_vel_1, ang_vel_2, ang_vel_3
### DEPENDENTS: a,   e,   i,   omega, RAAN, theta, r,  lat, lon, euler3, euler1,    euler3

states_array = result2array(simulator.state_history)
dependents_array = result2array(simulator.dependent_variable_history)
epochs = states_array[:,0] / constants.JULIAN_DAY
time_label = 'Time since J2000 [days]'
# gravity_field = field_type + ', ' + field_source

# run_checks(bodies, states_array, dependents_array, [0, 0, 0, 0, 1, 1], title_addition = '(undamped dynamics, ' + gravity_field + ')')

# plt.figure()
# plt.plot(epochs, dependents_array[:,1] / 1e3)
# plt.grid()
# plt.xlabel(time_label)
# plt.ylabel(r'$a$ [km]')
# plt.title('Semimajor axis')
#
# plt.figure()
# plt.plot(epochs, dependents_array[:,2])
# plt.grid()
# plt.xlabel(time_label)
# plt.ylabel(r'$e$ [-]')
# plt.title('Eccentricity')
#
# plt.figure()
# plt.plot(epochs, np.degrees(dependents_array[:,3]), label = r'$i$')
# plt.plot(epochs, np.degrees(dependents_array[:,4]), label = r'$\omega$')
# plt.plot(epochs, np.degrees(dependents_array[:,5]), label = r'$\Omega$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Orbit\'s Euler angles')
#
# plt.figure()
# plt.plot(epochs, np.degrees(dependents_array[:,9]), label = r'Lon')
# plt.plot(epochs, np.degrees(dependents_array[:,8]), label = r'Lat')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Coordinate [º]')
# plt.title('Mars\' coordinates in Phobos\' sky')
#
# cmap = plt.get_cmap('PRGn')
# fig, axis = plt.subplots()
# axis.scatter(np.degrees(dependents_array[:,9]), np.degrees(dependents_array[:,8]), c = epochs, cmap = cmap)
# axis.grid()
# axis.set_xlabel('Longitude [º]')
# axis.set_ylabel('Latitude [º]')
# axis.set_title('Mars\' coordinates in Phobos\' sky')
# fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs[0], epochs[-1])), ax=axis,
#                  orientation='vertical', label=time_label)
#
# plt.figure()
# plt.plot(epochs, np.degrees(bring_inside_bounds(dependents_array[:,10], 0.0, TWOPI)), label = r'$\Psi$')
# plt.plot(epochs, np.degrees(dependents_array[:,11]), label = r'$\theta$')
# plt.plot(epochs, np.degrees(dependents_array[:,12]), label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Phobos\' Euler angles')
#
# plt.figure()
# plt.plot(epochs, np.degrees(bring_inside_bounds(dependents_array[:,13], 0.0, TWOPI)), label = r'$\Psi$')
# plt.plot(epochs, np.degrees(dependents_array[:,14]), label = r'$\theta$')
# plt.plot(epochs, np.degrees(dependents_array[:,15]), label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Phobos\' Euler angles w.r.t. the Martian equator')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                DAMPING

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

phobos_mean_rotational_rate = 0.000228035245  # In rad/s
# As dissipation times, we will start with 4h and keep duplicating the damping time in each iteration. In the final
# iteration, a damping time of 2048h means a propagation time of 20480h, which is a bit over 2 years.
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0])*3600.0)
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         combined_propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)

damped_state_history = damping_results.forward_backward_states[-1][1]
damped_dependent_variable_history = damping_results.forward_backward_dependent_variables[-1][1]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

                                                PLOTS

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Here, we reuse the "epochs" array from before
damped_states_array = result2array(damping_results.forward_backward_states[-1][1])[:len(epochs)]
damped_dependents_array = result2array(damping_results.forward_backward_dependent_variables[-1][1])[:len(epochs)]

# plt.figure()
# plt.plot(epochs, damped_dependents_array[:,1] / 1e3)
# plt.grid()
# plt.xlabel(time_label)
# plt.ylabel(r'$a$ [km]')
# plt.title('Semimajor axis (damped dynamics)')
#
# plt.figure()
# plt.plot(epochs, damped_dependents_array[:,2])
# plt.grid()
# plt.xlabel(time_label)
# plt.ylabel(r'$e$ [-]')
# plt.title('Eccentricity (damped dynamics)')
#
# plt.figure()
# plt.plot(epochs, np.degrees(damped_dependents_array[:,3]), label = r'$i$')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,4]), label = r'$\omega$')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,5]), label = r'$\Omega$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Orbit\'s Euler angles (damped dynamics)')
#
# plt.figure()
# plt.plot(epochs, np.degrees(damped_dependents_array[:,9]), label = r'Lon')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,8]), label = r'Lat')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Coordinate [º]')
# plt.title('Mars\' coordinates in Phobos\' sky (damped dynamics)')
#
# cmap = plt.get_cmap('PRGn')
# fig, axis = plt.subplots()
# axis.scatter(np.degrees(damped_dependents_array[:,9]), np.degrees(damped_dependents_array[:,8]), c = epochs, cmap = cmap)
# axis.grid()
# axis.set_xlabel('Longitude [º]')
# axis.set_ylabel('Latitude [º]')
# axis.set_title('Mars\' coordinates in Phobos\' sky (damped dynamics)')
# fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs[0], epochs[-1])), ax=axis,
#                  orientation='vertical', label=time_label)
#
# plt.figure()
# plt.plot(epochs, np.degrees(bring_inside_bounds(damped_dependents_array[:,10], 0.0, TWOPI)), label = r'$\Psi$')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,11]), label = r'$\theta$')
# plt.plot(epochs, np.degrees(damped_dependents_array[:,12]), label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Phobos\' Euler angles (damped dynamics)')
#
# plt.figure()
# plt.plot(epochs, np.degrees(bring_inside_bounds(damped_dependents_array[:,13], -PI, PI, 'upper')), label = r'$\Psi$')
# plt.plot(epochs, np.degrees(bring_inside_bounds(damped_dependents_array[:,14], -PI, PI, 'upper')), label = r'$\theta$')
# plt.plot(epochs, np.degrees(bring_inside_bounds(damped_dependents_array[:,15], -PI, PI, 'upper')), label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel(time_label)
# plt.ylabel(r'Angle [º]')
# plt.title('Phobos\' Euler angles w.r.t. the Martian equator (damped dynamics)')
#
# plt.show()

for iter in range(1,len(dissipation_times)-1):
    run_checks(bodies,
               result2array(damping_results.forward_backward_states[iter][1]),
               result2array(damping_results.forward_backward_dependent_variables[iter][1]),
               [0, 0, 0, 0, 1, 1],
               title_addition = '(iteration ' + str(iter) + ', ' + gravity_field + ')')

run_checks(bodies, damped_states_array, damped_dependents_array, [0, 0, 0, 0, 1, 1], title_addition = '(damped dynamics, ' + gravity_field + ')')

# average_mean_motion = np.array([np.mean(result2array(iteration_dependents[-1][1])[:,16]) for iteration_dependents in \
#                                 damping_results.forward_backward_dependent_variables])
# iterations = np.array(list(range(len(dissipation_times)+1)))
# plt.figure()
# plt.plot(iterations, average_mean_motion * 86400.0, marker = '.')
# plt.axhline(0.0002278563609852602*86400.0, label = 'Original', ls = '--', c = 'r')
# plt.grid()
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel(r'$\bar n$ [rad/day]')
# plt.title('Evolution of average mean motion')

print('\n\nPROGRAM FINISHED SUCCESSFULLY')