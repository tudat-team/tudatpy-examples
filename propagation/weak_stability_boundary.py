"""
# Weak Stability Boundary demonstartion
## Objectives
 
This examples shows how to use **TUDAT functionalities** to:
1) Define a custom spacecraft / object function to be analyzed
2) Simulate N-body non-keplerian dynamics
3) Simulate complex dynamics of Laplace - coupled Jovian system
4) propagate using a **variable time step**
5) Postprocess and visualize results. 

"""

"""
## Import Statements
To start off, we import all the relevant modules.
"""
#standard imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple
from scipy.signal import argrelextrema
from scipy.optimize import brentq
# Tudat imports
import tudatpy
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import frame_conversion
from matplotlib.animation import FuncAnimation

#loading spice kernels:
spice.load_standard_kernels()  

"""
# Wrapper Classes
We define **two classes** to be used throughout the code:
1) `OrbitalSate`

   This is a wrapper class used for the initial state of the object of interest. Initialization takes Orbital elements:

- SMA => Semi-major axis                (m)
- ECC => Eccentricity                   (-)
- INC => Inclination                    (deg)
- LPE => Longitude of Periapsis         (deg)
- LAN => Longitude of Ascending Node    (deg)
- MNA => Mean Anomaly                   (rad)

3) `ReferenceFrame`
Enum class; wrapping Jovian system bodies to indices:


- Sun       =>  0
- Jupiter   =>  1
- Callisto  =>  2
- Ganymedde =>  3
- Europa    =>  4
- Io        =>  5
"""
#helpers to define the initial orbit of the body of interest:
class OrbitalSate:
    def __init__(self, SMA, ECC, INC, LPE, LAN, MNA):
        self.eccentricity       = ECC
        self.semi_major_axis    = SMA
        self.lan                = np.deg2rad(LAN)
        self.arg_of_periapsis   = np.deg2rad(LPE)
        self.inclination        = np.deg2rad(INC)
        self.mean_anomaly       = MNA

class ReferenceFrame:
    def __init__(self):
        self.Sun        = 0
        self.Jupiter    = 1
        self.Callisto   = 2
        self.Ganymede   = 3
        self.Europa     = 4
        self.Io         = 5





"""
# Auxiliary Functions
We define **two auxiliary functions** to be used throughout the code:
1) `spacecraft_state_function`

   This is the custom ephemeris function that defines the initial state of the custom body. The function takes as an input:
   - current_time (required to set up custom epehmeris)
   - object_state => OrbitalState class object with orbital information.

   The function then converts the OrbitalState parameters into cartesian position and velocity components in the J2000 frame.


2) `create_bodies`
    This function takes as inputs:


    - Frame
    - spacecraft / object name
    - initial state of the object

    and it creates the Jovian System + Sun + Object bodies.
3) `initialize_simulation`
    This function takes as inputs:


    - Frame
    - spacecraft / object name
    - initial state of the object

    and it creates the Jovian System + Sun + Object bodies.
"""

def spacecraft_state_function(current_time, object_state = None):
    """
    Custom ephemeris function for the spacecraft/object body.

    Converts a set of classical (Keplerian) orbital elements into a
    Cartesian state vector [x, y, z, vx, vy, vz].

    Steps:
      1. Wraps the mean anomaly M to [-pi, pi] and solves Kepler's
         equation (M = E - e*sin(E)) for the eccentric anomaly E via
         Newton-Raphson iteration (elliptic orbits only, e < 1).
      2. Computes position/velocity in the perifocal (orbital-plane)
         frame from E, the semi-major axis, and Ganymede's
         gravitational parameter.
      3. Builds the P/Q rotation vectors from the orbital elements
         (LAN, argument of periapsis, inclination) to rotate the
         perifocal state into the equatorial inertial frame.
      4. Applies an additional rotation (Rx_eq, using an obliquity
         angle of ~25.4393 deg) to express the state in the target
         reference frame.
    """
    eccentricity        = object_state.eccentricity
    semi_major_axis     = object_state.semi_major_axis
    lan                 = object_state.lan
    arg_of_periapsis    = object_state.arg_of_periapsis
    inclination         = object_state.inclination
    M                   = object_state.mean_anomaly

    if eccentricity < 1.0:

        M = (M + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi] for elliptic orbit
        #solve M = E - e*sin(E)
        E = M if eccentricity < 0.8 else np.pi
        for _ in range(100):
            delta = (E - eccentricity * np.sin(E) - M) / (1.0 - eccentricity * np.cos(E))
            E -= delta
            if abs(delta) < 1e-11:
                break
        cos_E = np.cos(E)
        sin_E = np.sin(E)

        # Perifocal position & velocity
        r = semi_major_axis * (1.0 - eccentricity * cos_E)
        x_perifocal  =  semi_major_axis * (cos_E - eccentricity)
        y_perifocal  =  semi_major_axis * np.sqrt(1.0 - eccentricity**2) * sin_E
        # 9887819980080.977 => Ganymede grav parameter (spice.get_body_gravitational_parameter("Ganymede"))
        v_factor     =  np.sqrt(9887819980080.977 * semi_major_axis) / r
        vx_perifocal =  v_factor * (-sin_E)
        vy_perifocal =  v_factor * np.sqrt(1.0 - eccentricity**2) * cos_E

    # rotate into inertial frame
    cos_o, sin_o = np.cos(lan),              np.sin(lan)
    cos_w, sin_w = np.cos(arg_of_periapsis), np.sin(arg_of_periapsis)
    cos_i, sin_i = np.cos(inclination),      np.sin(inclination)

    P = np.array([
         cos_o * cos_w - sin_o * sin_w * cos_i,
         sin_o * cos_w + cos_o * sin_w * cos_i,
         sin_w * sin_i
    ])
    Q = np.array([
        -cos_o * sin_w - sin_o * cos_w * cos_i,
        -sin_o * sin_w + cos_o * cos_w * cos_i,
         cos_w * sin_i
    ])

    position_3d = x_perifocal * P + y_perifocal * Q
    velocity_3d = vx_perifocal * P + vy_perifocal * Q
    epsilon = np.deg2rad(25.4392911)  

    Rx_eq = np.array([
        [1,              0,               0],
        [0,  np.cos(epsilon), -np.sin(epsilon)],
        [0,  np.sin(epsilon),  np.cos(epsilon)]
    ])

    position_3d = Rx_eq @ (x_perifocal * P + y_perifocal * Q)
    velocity_3d = Rx_eq @ (vx_perifocal * P + vy_perifocal * Q)

    return np.concatenate([position_3d, velocity_3d])

def create_bodies(FRAME = "J2000", spacecraft="target", initial_state = None):
    """
    Assembles the tudat SystemOfBodies for the simulation.

    - Creates default body settings for the Jovian system (Sun,
      Jupiter, Callisto, Ganymede, Europa, Io) in the given global
      frame (origin SSB, orientation FRAME).
    - Adds an empty body for the spacecraft/object and assigns it a
      custom ephemeris (spacecraft_state_function) that computes its
      state relative to Ganymede.
    - Builds and returns the SystemOfBodies object, along with the
      updated list of body names (including the spacecraft/object) and the
      BodySettings object used to create it.

    Returns:
        bodies (SystemOfBodies), bodies_to_create (list[str]),
        body_settings (BodyListSettings)
    """
    bodies_to_create = ["Sun", "Jupiter", "Callisto", "Ganymede", "Europa", "Io"]

    global_frame_origin = "SSB"        
    global_frame_orientation = FRAME


    bodies = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation
    )
    body_settings.add_empty_settings(spacecraft )
    body_settings.get( spacecraft ).ephemeris_settings =  environment_setup.ephemeris.custom_ephemeris( 
        lambda t: spacecraft_state_function(t, initial_state),
        'Ganymede', 
        FRAME,
    )
    
    #neglect the gravitational field of the object
    body_settings.get(spacecraft).gravity_field_settings = (
        environment_setup.gravity_field.central(0.0)
    )

    body_settings.get("Ganymede").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Jupiter", global_frame_orientation, "IAU_" + "Ganymede")

    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies_to_create.append(spacecraft)
    return bodies, bodies_to_create, body_settings

def initialize_simulation(bodies, bodies_to_create, body_settings, simulation_start_epoch, simulation_end_epoch):
    """
    Builds the acceleration models and initial conditions for the simulation.

    - Jupiter and Moons' acceleration model is set to spherical harmonic (5, 5)
    - Sun's and Body's acceleration model is set to point mass gravity
    - The central body of propagation for every body to the
      solar system is set as the barycenter (SSB).
    - Creates termination settings that stop the propagation at
      simulation_end_epoch.

    Returns:
        system_initial_state, central_bodies, acceleration_models,
        termination_settings
    """
    #create acceleration models
    acceleration_dict = {}
    for body_i in bodies_to_create:
        current_accelerations = {}
        for body_j in bodies_to_create:
            if body_i != body_j:
                if body_j != "Sun" and body_j != "target":
                    current_accelerations[body_j] = [
                        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
                    ]
                else:
                     current_accelerations[body_j] = [
                        propagation_setup.acceleration.point_mass_gravity()
                    ]
        acceleration_dict[body_i] = current_accelerations
    

  
    central_bodies = ["SSB"] * len(bodies_to_create)

    acceleration_models = propagation_setup.create_acceleration_models(
        body_system=bodies,
        selected_acceleration_per_body=acceleration_dict,
        bodies_to_propagate=bodies_to_create,
        central_bodies=central_bodies
    )

    system_initial_state = propagation.get_initial_state_of_bodies(
        bodies_to_propagate=bodies_to_create,
        central_bodies=central_bodies,
        body_system=bodies,
        initial_time=simulation_start_epoch
    )


    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    return system_initial_state, central_bodies, acceleration_models,  termination_settings

def simulate(bodies, bodies_to_create, system_initial_state, central_bodies, acceleration_models,  termination_settings , simulation_start_epoch, integrator = "fixed", fixed_step_size = 100):
    """
    Configures the integrator and propagator, runs the numerical
    propagation, and returns the resulting state history.

    - If integrator == "fixed": uses a fixed-step RKF7(8) integrator
    - Otherwise: uses a variable-step RKF4(5) integrator with
      elementwise scalar tolerance-based step-size control
      (abs/rel tol = 1e-10) and step size bounded between 0.001 s
      and 1000 s, starting from a 30 s initial step.
    - Builds Cowell-formulation translational propagator settings
      from the central bodies, acceleration models, propagated
      bodies, initial state, start epoch, integrator settings, and
      termination settings.

    Returns:
        system_state_array (np.ndarray), rotation_matrix (3x3 np.ndarray)
    """

    if integrator == "fixed":
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            time_step = fixed_step_size,
            coefficient_set =  propagation_setup.integrator.CoefficientSets.rkf_78,
            order_to_use =  propagation_setup.integrator.OrderToIntegrate.higher 
        )
    else:
         # Create RK4(5) settings
        control_settings =  propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance( 1.0E-10, 1.0E-10 )
        validation_settings =  propagation_setup.integrator.step_size_validation( 0.001, 1000.0 )
        integrator_settings =  propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step = 30.0,
            coefficient_set =  propagation_setup.integrator.CoefficientSets.rkf_45,
            step_size_control_settings = control_settings,
            step_size_validation_settings = validation_settings 
        )
    propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_create,
            system_initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_settings,
            propagation_setup.propagator.cowell
    )

    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, 
        propagator_settings
    )

    system_state_array = result2array(dynamics_simulator.state_history)

    #return a rotation matrix for conversion from J2000 into ECLIPJ2000
    rotation_matrix = spice.compute_rotation_matrix_between_frames(
        "J2000",       # equatorial J2000
        "ECLIPJ2000",  # ecliptic J2000
        0.0           
    )

    rotation_matrix = spice.compute_rotation_matrix_between_frames("J2000", "ECLIPJ2000", 0.0) 
    return system_state_array, rotation_matrix

"""
# Plotting Functions
We define **three plotting functions** to be used throughout the code:
1) `plot_jovian_system`

   This function takes as inputs:


    - central_body          => position array of the central body 
    - moon                  => position array of the target moon
    - center                => central body of the IRF
    - system_state_array    => TUDat propogator ourput - evolution of body positions


2) `plot_spacecraft_trajectories`
    This function takes as inputs:


    - begin                 => begin index of the state array (can be used for slicing the array)
    - end                   => end index of the state array (can be used for slicing the array)
    - center                => central body of the IRF
    - system_state_array    => TUDat propogator ourput - evolution of body positions

    And plots the Spacecraft/ Body trajectories in the chosen frame.
3) `plot_spacecraft_trajectories_triple`
    This function takes as inputs:


    - begin                 => begin index of the state array (can be used for slicing the array)
    - end                   => end index of the state array (can be used for slicing the array)
    - center                => central body of the IRF
    - system_state_array    => TUDat propogator ourput - evolution of body positions

    And plots the Jovian system + Distance from Jupiter + Distance from Ganymede in a triple window fashion.
4) `plot_mission_segment`
    This function acts as a wrapper for making multiple plots / calls of previously defined plotting functions:


    - begin                 => begin index of the state array (can be used for slicing the array)
    - end                   => end index of the state array (can be used for slicing the array)
    - center                => central body of the IRF
    - system_state_array    => TUDat propogator ourput - evolution of body positions

"""
def plot_jovian_system(system_state_array, central_body, moon, begin = 0, end = None, skip_moons=False, ax1=None, fig1=None, sphere=True, normalization=1, cube=1, frame_name = "Jupiter"):
    """
    If ax1 is provided, draws into that (existing) 3D axis instead of
    creating a new standalone figure — lets this be used as one panel
    of a larger subplot layout.
    """
    end = len(system_state_array[:, 0]) if end is None else end
    standalone = ax1 is None
    if standalone:
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(111, projection='3d')

    ax1.set_title(f"{frame_name}-Fixed Inertial Reference Frame")

    if not skip_moons:
        for i in range(len(bodies_to_create)):
            if i == 0 or i == len(bodies_to_create) - 1:
                continue
            body_state = system_state_array[begin:end, i * 6 + 1 : i * 6 + 4]
            body_state = body_state - central_body[begin:end, :3]
            state = (rotation_matrix @ body_state.T).T
            ax1.plot(state[:, 0] / normalization, state[:, 1]/ normalization, state[:, 2]/ normalization, label=bodies_to_create[i])
            ax1.scatter(state[0, 0]/ normalization, state[0, 1]/ normalization, state[0, 2]/ normalization)

    # sphere at the origin
    if sphere:
        sphere_radius = 1
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='orange', alpha=0.6, zorder=0)

    ax1.set_box_aspect([1, 1, 1])
    ax1.scatter(0, 0, 0)
    ax1.legend()
    target_math = frame_name.replace(" ", r"\ ")  

    ax1.set_xlabel(rf"$x / R_{{\mathrm{{{target_math}}}}}$")
    ax1.set_xlim([-cube, cube])
    ax1.set_ylabel(rf"$x / R_{{\mathrm{{{target_math}}}}}$")
    ax1.set_ylim([-cube, cube])
    ax1.set_zlabel(rf"$x / R_{{\mathrm{{{target_math}}}}}$")
    ax1.set_zlim([-cube, cube])

    if standalone:
        plt.tight_layout()
    return fig1, ax1

def plot_spacecraft_trajectories(begin, end, center, moon, system_state_array, ax1, ax2=None, color="blue", moon_state=None, normalization=1, target="Ganymede", cube=1):
    """
    ax1: 3D axis for the trajectory plot
    ax2: optional 2D axis; if given, also plots distance-from-Ganymede
         vs time for this segment, with apoapsis/periapsis marked.
    moon_state: defaults to the global `moon` (Ganymede) state array.
    """
    if moon_state is None:
        moon_state = moon

    for i in range(len(bodies_to_create)):
        if i == 0:
            continue
        if i == len(bodies_to_create) - 1:
            sc_pos_raw  = system_state_array[begin:end, i * 6 + 1 : i * 6 + 4]
            body_state  = sc_pos_raw - center[begin:end, :3]
            state = (rotation_matrix @ body_state.T).T
            state[:, :3] /= normalization
            ax1.plot(state[:, 0], state[:, 1], state[:, 2], label="Spacecraft", color=color)
            ax1.scatter(state[0, 0], state[0, 1], state[0, 2])

            time = system_state_array[begin:end, 0]
            time = (time - system_state_array[0, 0]) / (60 * 60 * 24)

            
            sep  = body_state
            dist = np.sqrt(sep[:, 0] ** 2 + sep[:, 1] ** 2 + sep[:, 2] ** 2) / normalization

            apogee_idx  = argrelextrema(dist, np.greater, order=10)[0]
            perigee_idx = argrelextrema(dist, np.less,    order=10)[0]

            # mark on the 3D trajectory
            for idx in apogee_idx:
                ax1.scatter(state[idx, 0], state[idx, 1], state[idx, 2],
                            color="red", marker='^', s=60)
                ax1.text(state[idx, 0], state[idx, 1], state[idx, 2],
                          'apoapsis', fontsize=7, color="red")
            for idx in perigee_idx:
                ax1.scatter(state[idx, 0], state[idx, 1], state[idx, 2],
                            color="red", marker='v', s=60)
                ax1.text(state[idx, 0], state[idx, 1], state[idx, 2],
                          'periapsis', fontsize=7, color="grey")

            # side-by-side distance-vs-time plot
            if ax2 is not None:
                ax2.plot(time, dist, color=color)
                if len(apogee_idx):
                    ax2.scatter(time[apogee_idx], dist[apogee_idx],
                                color="red", marker='^', s=60, label="apoapsis")
                if len(perigee_idx):
                    ax2.scatter(time[perigee_idx], dist[perigee_idx],
                                color="grey", marker='v', s=60, label="periapsis")
                ax2.set_xlabel('time [days]')
                target_math = target.replace(" ", r"\ ")  # escape spaces for mathtext
                ax2.set_title(rf"Distance from {target}")
                ax2.set_ylabel(rf"Distance from {target} [$(r / R_{{\mathrm{{{target_math}}}}})$]")
                ax2.legend()

def plot_spacecraft_trajectories_triple(begin, end, center, system_state_array, ax1, ax2=None, ax3=None, color="blue",
                                          moon_state=None,
                                          normalization = 1,
                                          cube = 1,
                                          extra_moons=(("Ganymede", "seagreen"), )):
    """
    Same as plot_spacecraft_trajectories, but with an additional third axis
    (ax3) showing distance from a set of moons (e.g. Ganymede and Europa)
    overlaid on the same shared y-scale.

    ax1: 3D axis for the trajectory plot
    ax2: optional 2D axis; plots distance-from-`center` vs time for this
         segment, with apoapsis/periapsis marked (center-relative).
    ax3: optional 2D axis; plots distance from each moon in `extra_moons`
         vs time, each with its own apoapsis/periapsis markers.
    moon_state: defaults to the global `moon` (Ganymede) state array,
        used for the ax2 apoapsis/periapsis markers on the 3D plot.
    extra_moons: iterable of (moon_name, color) pairs, where moon_name is
        an attribute on ReferenceFrame (e.g. "Ganymede", "Europa"). Each
        is plotted as its own distance-vs-time curve on ax3.
    """
    if moon_state is None:
        moon_state = moon

    ref = ReferenceFrame()

    for i in range(len(bodies_to_create)):
        if i == 0:
            continue
        if i == len(bodies_to_create) - 1:
            sc_pos_raw  = system_state_array[begin:end, i * 6 + 1 : i * 6 + 4]
            body_state  = sc_pos_raw - center[begin:end, :3]
            state = (rotation_matrix @ body_state.T).T
            state[:, :3] /= normalization

            ax1.plot(state[:, 0], state[:, 1], state[:, 2], label="JUICE", color=color)
            ax1.scatter(state[0, 0], state[0, 1], state[0, 2])

            time = system_state_array[begin:end, 0]

            # distance from `center` (e.g. Jupiter), used for ax1 markers + ax2
            sep  = body_state
            dist = np.sqrt(sep[:, 0] ** 2 + sep[:, 1] ** 2 + sep[:, 2] ** 2)

            apogee_idx  = argrelextrema(dist, np.greater, order=10)[0]
            perigee_idx = argrelextrema(dist, np.less,    order=10)[0]

            time = system_state_array[begin:end, 0]
            time = (time - system_state_array[0, 0]) / (60 * 60 * 24)
            # mark on the 3D trajectory
            for idx in apogee_idx:
                ax1.scatter(state[idx, 0], state[idx, 1], state[idx, 2],
                            color="red", marker='^', s=60)
                ax1.text(state[idx, 0], state[idx, 1], state[idx, 2],
                          'apoapsis', fontsize=7, color="red")
            for idx in perigee_idx:
                ax1.scatter(state[idx, 0], state[idx, 1], state[idx, 2],
                            color="red", marker='v', s=60)
                ax1.text(state[idx, 0], state[idx, 1], state[idx, 2],
                          'periapsis', fontsize=7, color="grey")

            # side-by-side distance-from-center plot
            if ax2 is not None:
                ax2.plot(time, dist, color=color, label="distance")
                if len(apogee_idx):
                    ax2.scatter(time[apogee_idx], dist[apogee_idx],
                                color="red", marker='^', s=60, label="apoapsis")
                if len(perigee_idx):
                    ax2.scatter(time[perigee_idx], dist[perigee_idx],
                                color="grey", marker='v', s=60, label="periapsis")
                ax2.set_xlabel('time [days]')
                ax2.set_ylabel('distance from Jupiter [m]')
                ax2.set_title('Distance from Jupiter [m]')
                #ax2.legend()

            # third plot: distance from each moon in extra_moons
            if ax3 is not None:
                for moon_name, moon_color in extra_moons:
                    moon_idx = getattr(ref, moon_name)
                    moon_pos = system_state_array[begin:end,
                                                   moon_idx * 6 + 1 : moon_idx * 6 + 4]
                    m_sep  = sc_pos_raw - moon_pos
                    m_dist = np.sqrt(m_sep[:, 0] ** 2 + m_sep[:, 1] ** 2 + m_sep[:, 2] ** 2)

                    m_apo_idx = argrelextrema(m_dist, np.greater, order=10)[0]
                    m_peri_idx = argrelextrema(m_dist, np.less,    order=10)[0]

                    ax3.plot(time, m_dist, color=moon_color, label=f"distance to {moon_name}")
                    if len(m_apo_idx):
                        ax3.scatter(time[m_apo_idx], m_dist[m_apo_idx],
                                    color="red", marker='^', s=60)
                    if len(m_peri_idx):
                        ax3.scatter(time[m_peri_idx], m_dist[m_peri_idx],
                                    color="red", marker='v', s=60)

                ax3.set_xlabel('time [s]')
                ax3.set_ylabel('distance [m]')
                ax3.set_title('Distance from Ganymede')
                #ax3.legend()

def plot_mission_segment(begin, end, system_state_array, color="blue", show_closeup = False):
    CENTRAL_BODY    = ReferenceFrame().Jupiter
    TARGET_MOON     = ReferenceFrame().Ganymede
    central_body    = system_state_array[:, CENTRAL_BODY * 6 + 1    : CENTRAL_BODY * 6 + 7]
    moon            = system_state_array[:, TARGET_MOON  * 6 + 1    : TARGET_MOON  * 6 + 7]
    spacecraft      = system_state_array[:, 6 * 6 + 1 : 6 * 6 + 7]
    times           = system_state_array[:, 0]
    cube =  20
    fig1 = plt.figure(figsize=(14, 7))
    ax1 = fig1.add_subplot(121, projection='3d')
    ax2 = fig1.add_subplot(122)
    plot_jovian_system(system_state_array=system_state_array, central_body=central_body, moon=moon, skip_moons=False, ax1=ax1, fig1=fig1, normalization=JUPITER_RADIUS, cube=cube)
    plot_spacecraft_trajectories(begin, end, system_state_array=system_state_array, center=central_body, moon=moon, ax1=ax1, ax2=ax2, color=color, target="Jupiter", normalization=JUPITER_RADIUS, cube=cube)
    plt.tight_layout()
    plt.show()


    #this can also be done in Ganymede-centered IRF:
    if show_closeup:
         
        CENTRAL_BODY    = ReferenceFrame().Ganymede
        TARGET_MOON     = ReferenceFrame().Ganymede
        central_body    = system_state_array[:, CENTRAL_BODY * 6 + 1    : CENTRAL_BODY * 6 + 7]
        moon            = system_state_array[:, TARGET_MOON  * 6 + 1    : TARGET_MOON  * 6 + 7]
        spacecraft      = system_state_array[:, 6 * 6 + 1 : 6 * 6 + 7]
        times           = system_state_array[:, 0]
        cube = 10

        fig1 = plt.figure(figsize=(14, 7))
        ax1 = fig1.add_subplot(121, projection='3d')
        ax2 = fig1.add_subplot(122)
        plot_jovian_system(system_state_array=system_state_array,central_body=central_body, moon=moon, skip_moons=True, ax1=ax1, fig1=fig1, normalization=GANYMEDE_RADIUS, cube=cube, frame_name="Ganymede")
        plot_spacecraft_trajectories(begin, end, system_state_array=system_state_array, center=central_body, moon=moon, ax1=ax1, ax2=ax2, color="green", target="Ganymede", normalization=GANYMEDE_RADIUS, cube=cube)
        plt.tight_layout()
        plt.show()
 
"""
# CR3BP Functions
We define **two CR3BP functions** to be used throughout the code:
1) `_draw_rotating_potential_2d`

   The purpose of this function is to switch to a Co-Rotating Frame and plot the target object trajectories. This function takes as arguments:



    - central_body          => position array of the central body 
    - moon                  => position array of the target moon
    - spacecraft            => position array of the body of interest
    - times                 => array of discrete timesteps
    
    Optional arguments allow to customize the plotting:

    -offset                  => allows to choose the x y scales for the plotting rectangle (in semi-major-axis normalized coordinates)
    -range                   => allows to choose the zoom scale for the xy plot


2) `plot_rotating_potential_dual`
    This function is a wrapper for plotting side by side the Co-Rotating frame with different zoom levels
"""

def _draw_rotating_potential_2d(central_body, moon, spacecraft, times, ax,
                                 range=0.1, offset=(1, 0), show_potential=True,
                                 title=None, color="violet", plot_isopotential=False, label=None):
    """
    Computes the CR3BP rotating-frame potential and draws it as a 2D
    imshow (color = potential) into the given axis `ax`, along with the
    orbiter trajectory, Ganymede, and Lagrange point labels.
    """
    moon_position =  moon[:, :3] - central_body[:, :3]
    moon_velocity =  moon[0, 3:] - central_body[0, 3:]

    moon_velocity = (rotation_matrix @ moon_velocity.T).T
    moon_position = (rotation_matrix @ moon_position.T).T

    spacecraft_position     =   spacecraft[:, :3] - central_body[:, :3]
    spacecraft_velocity     =   spacecraft[0, 3:] - central_body[0, 3:]
    spacecraft_velocity = (rotation_matrix @ spacecraft_velocity.T).T
    spacecraft_position = (rotation_matrix @ spacecraft_position.T).T  - moon_position

    distances = np.sqrt(moon_position[:, 0] ** 2 + moon_position[:, 1] ** 2 + moon_position[:, 2] ** 2)
    perijove = np.min(distances)
    apojove  = np.max(distances)
    
    a = semi_major_axis# 0.5 * (apojove + perijove) 
    print(f"computed: {semi_major_axis} theoretical {a}")

    mu_J = bodies.get("Jupiter").gravitational_parameter
    mu_G = bodies.get("Ganymede").gravitational_parameter
    mu_total = mu_J + mu_G
    mu    = mu_G / mu_total

    x_J = -mu
    x_G =  1.0 - mu
    a *= x_G 

    mean_motion = np.sqrt(mu_total / a ** 3)
    v_char = mean_motion * a
    omega = np.sqrt(mu_total / a**3) 

    t0 = times[0]
    angle0 = np.arctan2(moon_position[0, 1], moon_position[0, 0])

    # compute rotation w.r.t. the J2000 ecliptic to account for the moon's inclination
    x, y, z = moon_position[0]
    r = np.hypot(x, y)
    a1, b1 = x / r, y / r

    phi = -np.arctan2(z, r)
    c, s = np.cos(phi), np.sin(phi)

    R_inclination = np.array([
        [c + (1 - c) * b1**2,     -(1 - c) * a1 * b1,   -a1 * s],
        [-(1 - c) * a1 * b1,        c + (1 - c) * a1**2, -b1 * s],
        [ a1 * s,                   b1 * s,              c    ]
    ])

    moon_position_initial        = R_inclination @ moon_position[0]
    moon_velocity           = R_inclination @ moon_velocity
    spacecraft_velocity     = R_inclination @ spacecraft_velocity
    spacecraft_position_initial  = R_inclination @ spacecraft_position[0]

    pos_body_rot = np.zeros_like(moon_position)
    # rotation into the CR3B co-rotating frame
    R = np.array([[ np.cos(angle0), np.sin(angle0), 0],
                  [-np.sin(angle0), np.cos(angle0), 0],
                  [ 0,             0,             1]])
    vel_gan_rot     = R @ moon_velocity
    vel_sc_rot      = R @ spacecraft_velocity
    pos_sc_rot      = R @ spacecraft_position[0]
    pos_gan_rot     = R @ moon_position[0]

    v_rot_sc = vel_sc_rot - omega * np.sqrt(np.dot(pos_sc_rot, pos_sc_rot)) * vel_gan_rot / np.linalg.norm(vel_gan_rot)

    for j, t in enumerate(times):
        angle = omega * (t - t0) + angle0
        R = np.array([[ np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [ 0,             0,             1]])
        pos_body_rot[j] = R @ (spacecraft_position[j]  / (a) )
        pos_body_rot[j] += np.array([1, 0, 0])

    Npts = 800

    x_min, x_max = -range + offset[0], range   + offset[0]
    y_min, y_max = -range  + offset[1], range  + offset[1]

    xs = np.linspace(x_min, x_max, Npts)
    ys = np.linspace(y_min, y_max, Npts)
    X, Y = np.meshgrid(xs, ys)

    r1 = np.sqrt((X - x_J)**2 + Y**2)   # distance to Jupiter
    r2 = np.sqrt((X - x_G)**2 + Y**2)   # distance to Ganymede

    U = -0.5 * (X**2 + Y**2) - (1 - mu) / r1 - mu / r2

    def omega1(xv, yv):
        r1 = np.sqrt((xv - x_J)**2 + yv**2)   # distance to Jupiter
        r2 = np.sqrt((xv - x_G)**2 + yv**2)   # distance to Ganymede
        return 0.5 * (xv ** 2 + yv ** 2) + (1 - mu) / r1 + mu / r2

    print(f"Jacobi for the SC:{omega1(pos_sc_rot[0] / a, pos_sc_rot[1] / a) * 2 - (np.sqrt(np.dot(v_rot_sc, v_rot_sc)) / v_char) ** 2} ")

    def dUdx(xval, mu):
        r1 = abs(xval + mu)
        r2 = abs(xval - (1 - mu))
        return (xval
                - (1 - mu) * np.sign(xval + mu) / r1**2
                - mu       * np.sign(xval - (1 - mu)) / r2**2)
    L1_x = brentq(dUdx, x_G - 0.1,  x_G - 1e-6, args=(mu,))
    L2_x = brentq(dUdx, x_G + 1e-6, x_G + 0.1,  args=(mu,))
    L3_x = brentq(dUdx, -1.5,        x_J - 1e-6, args=(mu,))
    lagrange_x = [L1_x, L2_x, L3_x, 0.5 - mu,        0.5 - mu       ]
    lagrange_y = [0,    0,    0,     np.sqrt(3) / 2,  -np.sqrt(3) / 2]
    lagrange_labels = ['L1', 'L2', 'L3', 'L4', 'L5']

    U_L1 = omega1(L1_x, 0)
    U_L2 = omega1(L2_x, 0)
    print(f"Jacobi for L1: {2 * U_L1}")
    print(f"Jacobi for L2: {2 * U_L2}")
    print(f"excess energy: {2 * U_L1 - (omega1(pos_sc_rot[0] / a, pos_sc_rot[1] / a) * 2 - (np.sqrt(np.dot(v_rot_sc, v_rot_sc)) / v_char) ** 2)} ")

    U_plot = np.clip(U, -5, -1.3)

    if show_potential:
        ax.imshow(
            U_plot,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='RdYlBu_r',
            interpolation='bilinear',
            aspect='equal'
        )
    if label is not None:
        ax.plot(pos_body_rot[:, 0], pos_body_rot[:, 1],
            color=color, linewidth=0.8, alpha=0.8, label=label)
    else:
        ax.plot(pos_body_rot[:, 0], pos_body_rot[:, 1],
            color=color, linewidth=0.8, alpha=0.8)
    if plot_isopotential:
        for lx, ly, name in zip(lagrange_x, lagrange_y, lagrange_labels):
            if x_min < lx < x_max and y_min < ly < y_max:
                ax.text(lx, ly, name, fontsize=10, color='green')

        ax.plot(x_G, 0, 'o', color='steelblue', markersize=8, zorder=5)
        ax.plot(x_J, 0, 'o', color='steelblue', markersize=32, zorder=5)

        ax.set_xlabel(r'$x / a_{\mathrm{Ganymede}}$')
        ax.set_ylabel(r'$y / a_{\mathrm{Ganymede}}$')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        levels_lines = np.linspace(np.percentile(U_plot, 40), U_plot.max(), 120)
        ax.contour(X, Y, U_plot, levels=levels_lines,
                colors='black', linewidths=0.4, alpha=0.5)
    if title:
        ax.set_title(title)


def plot_rotating_potential_dual(central_body, moon, spacecraft, times,
                                  close_range=0.1, close_offset=(1, 0),
                                  wide_range=1.5, wide_offset=(0, 0),
                                  ax1=None,
                                  ax2=None,
                                  show_potential=True, color="violet",
                                  plot_isopotential=False, label=None,
                                  suptitle=None):
    """
    Side-by-side CR3BP rotating-frame potential plots (2D imshow):
    a close-up around the moon, and a wide view of the whole
    Jupiter-Ganymede system.
    """

    _draw_rotating_potential_2d(central_body, moon, spacecraft, times=times, ax=ax1,
                                 range=close_range, offset=close_offset,
                                 show_potential=show_potential, title="Close-up Ganymede",
                                 color=color, plot_isopotential=plot_isopotential, label=label)
    _draw_rotating_potential_2d(central_body, moon, spacecraft, times=times, ax=ax2,
                                 range=wide_range, offset=wide_offset,
                                 show_potential=show_potential, title="Jupiter-Ganymede CR3BP system",
                                 color=color, plot_isopotential=plot_isopotential, label=label)

    if suptitle is not None:
        fig = ax1.figure if ax1 is not None else ax2.figure
        fig.suptitle(suptitle)






"""
## Environment Setup
### We Initiate spice kernels, simulation start and end dates, and create the system of bodies
As for our framework, we need to create bodies for: Europa, Ganymede, Callisto, Io Jupiter and the Sun.
We will select Solar System Barycenter (SSB) as origin of our global frame, and orient it according to the J2000. Next, the defualt body settings can be defined.
We also define an initial state for our object of interest. This has been chosen manualy to lie in the WSB of Ganymede:
"""

#initialize our object orbit around Ganymede
object = OrbitalSate(
    SMA = 12620671.675021729,
    ECC = 0.71771389645297878,
    INC = 0.41381573069655914,
    LPE = 223.98450368357575 ,
    LAN = 185.69309224626306,
    MNA = 6.1484626004311655,
)

#define simulation window - we start in 2036, similar to  JUICE mission and simulate for a year:
simulation_start_epoch = DateTime(2036, 1,27, 12,40,45).to_epoch()
simulation_end_epoch   = simulation_start_epoch + constants.JULIAN_YEAR
fixed_step_size = 2000 #fixed step size if we choose to use fixed step propogator


#we use the helper functions to create the bodies
bodies, bodies_to_create, body_settings = create_bodies(initial_state=object)

#some defines from SPICE to be used in plotting
JUPITER_RADIUS   = spice.get_average_radius("Jupiter") 
GANYMEDE_RADIUS  = spice.get_average_radius("Ganymede")

#initialize the simulation via the helper function. 
system_initial_state, central_bodies, acceleration_models,  termination_settings = initialize_simulation(bodies, bodies_to_create, body_settings, simulation_start_epoch, simulation_end_epoch)

#with the system initialized, create a simulation and simulate:
system_state_array, rotation_matrix = simulate(bodies, bodies_to_create, system_initial_state, central_bodies, acceleration_models, termination_settings, simulation_start_epoch, integrator="dynamic")


"""
## Analysis of the results
### Post-processing and visualization of the results. 
As for our framework, we need to create bodies for: Europa, Ganymede, Callisto, Io Jupiter and the Sun.
We will select Solar System Barycenter (SSB) as origin of our global frame, and orient it according to the J2000. Next, the defualt body settings can be defined.
"""




#We can now plot the evolution of the Jovian system:

EOM = len(system_state_array[:, 0]) #define the End of mission epoch

#use the helper function to plot for the duration of the simulation the evolution of the Jovian system + our body
plot_mission_segment(0, EOM, system_state_array=system_state_array)


"""
### Clearly, the full trajectory visualization looks a bit messy. This can be cleaned up by noticing that the simulated mission trajectory contains 4 key segments:
- initial weakly - stable orbit around Ganymede (at the WSB)
- Jupiter - centered orbit after escaping Ganymede
- second weak capture at Ganymede 
- Jupiter - centered orbit + perturbations from Ganymede flybys
These segments can be identified by considering the distance of the spacecraft from Ganymede:
"""


CENTRAL_BODY    = ReferenceFrame().Jupiter #we choose Jupiter as our central body for the Inertial Reference Frame
TARGET_MOON     = ReferenceFrame().Ganymede #Ganymede as our target Moon

#extract trajectory informations for CENTRAL_BODY, TARGET_MOON and spacecraft
central_body    = system_state_array[:, CENTRAL_BODY * 6 + 1    : CENTRAL_BODY * 6 + 7]
moon            = system_state_array[:, TARGET_MOON  * 6 + 1    : TARGET_MOON  * 6 + 7]
spacecraft      = system_state_array[:, 6 * 6 + 1 : 6 * 6 + 7]
times           = system_state_array[:, 0]

#get distance from Ganymede as a function of time (normalized by Ganymede radius):
spacecraft_distance_ganymede = spacecraft - moon #get distance from Ganymede
spacecraft_distance_ganymede = np.sqrt(
    spacecraft_distance_ganymede[:, 0] ** 2 + 
    spacecraft_distance_ganymede[:, 1] ** 2 + 
    spacecraft_distance_ganymede[:, 2] ** 2  
) / GANYMEDE_RADIUS


#now, find mission segments. We set the cutoff as 40 Ganymede radii for escape, and 34 Ganymede radii for capture:
moon_escape1    = np.argmax(spacecraft_distance_ganymede > 40)
moon_capture1   = moon_escape1 + np.argmax(spacecraft_distance_ganymede[moon_escape1:] < 34)
moon_escape2    = moon_capture1 + np.argmax(spacecraft_distance_ganymede[moon_capture1:] > 40)


#now plot them via helper functions. For WSB orbits around Ganymede, visualize that frame too:
plot_mission_segment(0, moon_escape1, system_state_array=system_state_array, show_closeup=True)
plot_mission_segment(moon_escape1, moon_capture1, system_state_array=system_state_array)
plot_mission_segment(moon_capture1, moon_escape2, system_state_array=system_state_array, show_closeup=True)


"""
###Plotting the Entire Trajectory
We can now once again switch back to Jupiter centered IRF and plot all of the trajectory segments there,
together with distances from Jupiter and Ganymede.

The results, including the ballistic pumpdown, are similar to the trajectories to be used by Juice during Callisto - Ganymede transfer.

https://link.springer.com/article/10.1007/s11214-024-01093-y
Pages 33 - 44
"""

#select Jupiter as the center
CENTRAL_BODY    = ReferenceFrame().Jupiter
TARGET_MOON     = ReferenceFrame().Ganymede
central_body    = system_state_array[:, CENTRAL_BODY * 6 + 1    : CENTRAL_BODY * 6 + 7]
moon            = system_state_array[:, TARGET_MOON * 6 + 1     : TARGET_MOON * 6 + 7]
sc = system_state_array[:, 6 * 6 + 1 : 6 * 6 + 7]
fig1 = plt.figure(figsize=(20, 7))
ax1 = fig1.add_subplot(131, projection='3d')
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)

#plot Jovian moon evolution
plot_jovian_system(system_state_array=system_state_array, central_body=central_body, moon=moon, skip_moons=False, ax1=ax1, fig1=fig1, normalization=JUPITER_RADIUS, cube=20)

#plot 1st mission segment (WSB orbit around Ganymede)
plot_spacecraft_trajectories_triple(0, moon_escape1, center=central_body,  system_state_array=system_state_array, ax1=ax1, ax2=ax2, ax3=ax3, color="blue", normalization=JUPITER_RADIUS, cube=20)

#plot 2nd mission segment (escape into Jupiter orbit)
plot_spacecraft_trajectories_triple(moon_escape1, moon_capture1, center=central_body, system_state_array=system_state_array, ax1=ax1, ax2=ax2, ax3=ax3, color="green", normalization=JUPITER_RADIUS, cube=20)

#plot 3rd mission segment (Weak capture at Ganymede)
plot_spacecraft_trajectories_triple(moon_capture1, moon_escape2, center=central_body, system_state_array=system_state_array, ax1=ax1, ax2=ax2, ax3=ax3, color="cyan", normalization=JUPITER_RADIUS, cube=20)

#plot 4th mission segment (escape into Jupiter orbit + pumpdown towards Europa)
plot_spacecraft_trajectories_triple(moon_escape2, len(system_state_array[:, 0]), center=central_body, system_state_array=system_state_array, ax1=ax1, ax2=ax2, ax3=ax3, normalization=JUPITER_RADIUS, color="violet", cube=20)
plt.tight_layout()
plt.show()

"""
###Plotting the Trajectory in Co-Rotating Frame
Better understanding of the trajectory can be gained by visualizing the trajectories in Jupiter-Ganymede Co-Rotating Frame.
The approach employed is that of a general CR3BP problem, see details in:
https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html

We use the helper functions for the CR3BP to plot the different mission segments 1 through 4:
"""


#we need to know the average semi-major axis of Ganymede at the current epoch. This can be computed from SPICE:

epoch = simulation_start_epoch
state = spice.get_body_cartesian_state_at_epoch(
    target_body_name="Ganymede",
    observer_body_name="Jupiter",
    reference_frame_name="ECLIPJ2000",
    aberration_corrections="NONE",
    ephemeris_time=epoch
)
jupiter_gm = spice.get_body_gravitational_parameter("Jupiter")
kepler_elements = element_conversion.cartesian_to_keplerian(state, jupiter_gm)

semi_major_axis = kepler_elements[0] 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

#plot the trajectories + Jacobi potential for CR3BP 
plot_rotating_potential_dual(
    central_body=central_body[0:moon_escape1],
    moon=moon[0:moon_escape1], 
    spacecraft=spacecraft[0:moon_escape1], 
    times=times[0:moon_escape1],
    close_range=0.1, 
    close_offset=(1, 0),
    wide_range=1.2, 
    wide_offset=(0, 0),
    show_potential=False,
    ax1=ax1,
    ax2=ax2,
    color="blue",
    plot_isopotential=True,
    label="Initial Weak-Orbit (Phase 1)",
    suptitle="Trajectories in Jupiter-Ganymede Co-Rotating Frame",
)
plot_rotating_potential_dual(
    central_body=central_body[moon_escape1:moon_capture1],
    moon=moon[moon_escape1:moon_capture1], 
    spacecraft=spacecraft[moon_escape1:moon_capture1], 
    times=times[moon_escape1:moon_capture1],
    close_range=0.1, 
    close_offset=(1, 0),
    wide_range=1.2, 
    wide_offset=(0, 0),
    show_potential=False,
    ax1=ax1,
    ax2=ax2,
    color="green",
    label="Jupiter centered orbit (Phase 2)"
)
plot_rotating_potential_dual(
    central_body=central_body[moon_capture1:moon_escape2],
    moon=moon[moon_capture1:moon_escape2], 
    spacecraft=spacecraft[moon_capture1:moon_escape2], 
    times=times[moon_capture1:moon_escape2],
    close_range=0.1, 
    close_offset=(1, 0),
    wide_range=1.2, 
    wide_offset=(0, 0),
    show_potential=True,
    ax1=ax1,
    ax2=ax2,
    color="cyan",
    label="Second Weak Capture (Phase 3)"
)
ax2.legend()
ax1.legend()
plt.tight_layout()
plt.show()

#finally, the rest of the trajectory evolution:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plot_rotating_potential_dual(
    central_body=central_body[moon_capture1:EOM],
    moon=moon[moon_capture1:EOM], 
    spacecraft=spacecraft[moon_capture1:EOM], 
    times=times[moon_capture1:EOM],
    close_range=0.1, 
    close_offset=(1, 0),
    wide_range=1.2, 
    wide_offset=(0, 0),
    show_potential=True,
    ax1=ax1,
    ax2=ax2,
    color="blue",
    plot_isopotential=True,
    label="Pumpdown towards Europa (Phase 4)",
    suptitle="Trajectories in Jupiter-Ganymede Co-Rotating Frame",
)
plt.tight_layout()
plt.show()
