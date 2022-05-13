import sys
sys.path.insert(0, "/cala/jeremie/tudat-bundle/build/tudatpy")

# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion


import time

t0 = time.time()

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = 0.0

# Create default body settings for "Earth"
bodies_to_create = ["Earth"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add vehicle object to system of bodies
bodies.create_empty_body("Delfi-C3")

# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on Delfi-C3
acceleration_settings_delfi_c3 = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Set initial conditions for the satellite that will be
# propagated in this simulation. The initial conditions are given in
# Keplerian elements and later on converted to Cartesian elements
initial_keplerian_elements = [7500e3, 0.1, np.deg2rad(85.3), np.deg2rad(235.7), np.deg2rad(23.4), np.deg2rad(139.87)]
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis=initial_keplerian_elements[0],
    eccentricity=initial_keplerian_elements[1],
    inclination=initial_keplerian_elements[2],
    argument_of_periapsis=initial_keplerian_elements[3],
    longitude_of_ascending_node=initial_keplerian_elements[4],
    true_anomaly=initial_keplerian_elements[5]
)
period = np.pi * np.sqrt(initial_keplerian_elements[0]**3/earth_gravitational_parameter)
simulation_end_epoch = 30*period # ~27 hours

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(
    simulation_end_epoch,
    terminate_exactly_on_final_condition=True)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    termination_condition
)

fixed_coefficients = [
    # propagation_setup.integrator.euler_forward,
    propagation_setup.integrator.rk_4,
    propagation_setup.integrator.explicit_mid_point,
    propagation_setup.integrator.explicit_trapezoid_rule,
    propagation_setup.integrator.ralston,
    propagation_setup.integrator.rk_3,
    propagation_setup.integrator.ralston_3,
    propagation_setup.integrator.SSPRK3,
    propagation_setup.integrator.ralston_4,
    propagation_setup.integrator.three_eight_rule_rk_4
]

variable_coefficients = [
    # propagation_setup.integrator.heun_euler,
    # propagation_setup.integrator.rkf_12,
    propagation_setup.integrator.rkf_45,
    propagation_setup.integrator.rkf_56,
    propagation_setup.integrator.rkf_78,
    propagation_setup.integrator.rkdp_87,
    propagation_setup.integrator.rkf_89,
    propagation_setup.integrator.rkv_89,
    propagation_setup.integrator.rkf_108,
    propagation_setup.integrator.rkf_1210,
    propagation_setup.integrator.rkf_1412
]

# to_use = "fixed"
# to_use = "variable"	 
# to_use = "variable (lower)"
# to_use = "variable (higher)"

variable_dts = np.logspace(2.5, 1.2, num=20)
variable_tols = np.logspace(-8, -2, num=20)

def get_dts(to_use, coeff_name):
    if to_use == "fixed":
        if coeff_name == "euler_forward":
            return np.logspace(-0.5, -2.5, num=20)
        return np.logspace(2.0, -0.75, num=20)
    else:
        if coeff_name == "rkf_1412":
            if to_use == "variable (lower)":
                return np.logspace(2.6, 1.85, num=20)
            return np.logspace(2.6, 1.5, num=20)
        return variable_dts

save_csv = [["Method", "Coefficients", "Slope"]]

for to_use in ["fixed", "variable (lower)", "variable (higher)"]:

    order_to_use = propagation_setup.integrator.OrderToIntegrate.lower
    if to_use == "fixed":
        integrators = fixed_coefficients
        title = "Fixed step size integrators"
    else:
        integrators = variable_coefficients
        title = "Variable step size integrators"
        if to_use == "variable (lower)":
            title = "Variable step size integrators (using lower order, fixed step size)"
        elif to_use == "variable (higher)":
            order_to_use = propagation_setup.integrator.OrderToIntegrate.higher
            title = "Variable step size integrators (using higher order, fixed step size)"


    plt.figure(figsize=(15, 9))
    # slopes, intercepts, colors, vals_range = [], [], [], []
    colors, x_fits, y_fits = [], [], []
    for i, coefficients in enumerate(integrators):
        list_of_f_evals, list_of_pos_error = [], []
        print(str(coefficients).split(".")[-1])
        for integ_param in get_dts(to_use, str(coefficients).split(".")[-1]):
            # Create numerical integrator settings
            if to_use == "variable":
                integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                    simulation_start_epoch,
                    1.0,
                    coefficients,
                    1e-10,
                    1e10,
                    integ_param,
                    integ_param,
                    maximum_factor_increase=1e10,
                    minimum_factor_increase=1e-10
                )
                param_print = "tol = %.2e" % integ_param
            else:
                integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                    simulation_start_epoch, integ_param, coefficients, order_to_use
                )
                param_print = "dt = %.2e s" % integ_param


            # Create simulation object and propagate the dynamics
            dynamics_simulator = numerical_simulation.SingleArcSimulator(
                bodies, integrator_settings, propagator_settings, print_state_data=False, print_dependent_variable_data=False
            )

            # Extract the resulting state history and convert it to an ndarray
            states = dynamics_simulator.state_history
            final_state = states[simulation_end_epoch]
            f_evals = list(dynamics_simulator.cumulative_number_of_function_evaluations.values())[-1]

            final_r_error = np.linalg.norm(final_state[0:3] - initial_state[0:3])
            if final_r_error < 2e-4:
                print("Skipping due to numerical accuracy (most likely) reached.")
                break
            list_of_f_evals.append(f_evals)
            list_of_pos_error.append(final_r_error)
            print("Final position error of %.2e m with %.2e f evals (%s)" % (final_r_error, f_evals, param_print))

        print("Plotting for this integrator...")
        if i == 0:
            p = plt.loglog(list_of_f_evals, list_of_pos_error, 'o-', color="black", label=str(coefficients).split(".")[-1])
        else:
            p = plt.loglog(list_of_f_evals, list_of_pos_error, 'o-', label=str(coefficients).split(".")[-1])

        f_evals_log, pos_error_log = np.log10(list_of_f_evals), np.log10(list_of_pos_error)

        fit_errors_median = 10
        print("Computing fit...")
        f_evals_range = [np.min(f_evals_log), np.max(f_evals_log)]
        for i_refine in range(2):
            log_slopes = []
            log_intercepts = []
            for j in range(len(f_evals_log)-1):
                log_slopes.append((pos_error_log[j+1]-pos_error_log[j])/(f_evals_log[j+1]-f_evals_log[j]))
                log_intercepts.append(pos_error_log[j]-log_slopes[-1]*f_evals_log[j])
                
            # Remove nan values
            log_slopes = np.array(log_slopes)
            log_intercepts = np.array(log_intercepts)
            idx_nan_slopes = np.isnan(log_slopes)
            idx_nan_intercepts = np.isnan(log_intercepts)
            idx_nan = np.logical_or(idx_nan_slopes, idx_nan_intercepts)
            log_slopes = log_slopes[~np.isnan(idx_nan)]
            log_intercepts = log_intercepts[~np.isnan(idx_nan)]

            # Compute the median of the slopes
            slope_median = np.median(log_slopes)
            idx_slope_median = np.argmin(np.abs(log_slopes-slope_median))
            intercept_median = log_intercepts[idx_slope_median]

            # Compute line of best fit
            x_fit = np.logspace(f_evals_range[0], f_evals_range[1], num=len(f_evals_log))
            y_fit = np.asarray([10**(slope_median*np.log10(_fevals)+intercept_median) for _fevals in x_fit])

            if i_refine == 0:
                # Compute difference between line of best fit and original values
                fit_errors = np.log10(np.fabs(np.asarray(10**pos_error_log)-np.asarray(y_fit)))
                fit_errors_median = np.fabs(np.median(fit_errors))
                # Only keep values below the median
                to_keep_idx = np.where(fit_errors <= fit_errors_median)[0]

                # Remove the points that are not within the median
                pos_error_log = pos_error_log[to_keep_idx]
                f_evals_log = f_evals_log[to_keep_idx]

        print(" > Slope for %s integrator of %.2f" % (str(coefficients).split(".")[-1], slope_median))
        save_csv.append([to_use, str(coefficients).split(".")[-1], slope_median])

        x_fits.append(x_fit), y_fits.append(y_fit)
        colors.append(p[0].get_color())
            
    print("Making final plot...")
    plt.legend()

    # Plot fitted lines
    # for x_fit, y_fit, color in zip(x_fits, y_fits, colors):
    #     plt.loglog(x_fit, y_fit, color=color, linestyle='--')

    plt.xlabel("Number of function evaluations")
    plt.ylabel("Final position error [m]")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig("/cala/jeremie/tudatpy-examples/propagation/%s.pdf" % to_use)
    plt.close()

# Save the results to a csv file
save_csv = np.asarray(save_csv)
np.savetxt("/cala/jeremie/tudatpy-examples/propagation/slopes.csv", save_csv, delimiter=",", fmt="%s")

print(time.time()-t0)