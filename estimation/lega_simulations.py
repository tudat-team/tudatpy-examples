# %%
from tudatpy.interface import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.astro.time_representation import DateTime
from datetime import datetime, timedelta
from tudatpy.dynamics import environment_setup, environment
from tudatpy.estimation import observable_models_setup, observations_setup
from tudatpy import estimation

import os
from tudatpy.math import interpolators
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %% [markdown]
# ## 2. Helper Functions
# %%
def strip_and_shift_first_column(input_file, output_folder, time_shift_seconds=0.4):
    """Removes the scan index and shifts the UTC time tag by the specified seconds."""
    filename = os.path.basename(input_file)
    output_path = os.path.join(output_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('#') or not line.strip():
                f_out.write(line)
            else:
                parts = line.split()
                if len(parts) >= 6:
                    utc_str = parts[1]
                    try:
                        dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%f")
                        dt_shifted = dt + timedelta(seconds=time_shift_seconds)
                        parts[1] = dt_shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    except ValueError:
                        dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S")
                        dt_shifted = dt + timedelta(seconds=time_shift_seconds)
                        parts[1] = dt_shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

                    f_out.write(" ".join(parts[1:]) + '\n')
    return output_path

# %% [markdown]
# ## 3. Simulation Setup

# %%
spice.load_standard_kernels()
spice.load_kernel("juice_archive/spk/juice_orbc_000097_230414_310721_v02.bsp")

# Define dates
start = datetime(2024, 8, 19); end = datetime(2024, 8, 21)
start_time = DateTime.from_python_datetime(start).to_epoch()
end_time = DateTime.from_python_datetime(end).to_epoch()
start_time_buffer = start_time - 86400; end_time_buffer = end_time + 86400

# Environment Setup
bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"; global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, start_time, end_time, global_frame_origin, global_frame_orientation)

body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006, global_frame_orientation,
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 10.0))
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

spacecraft_name = "JUICE"; spacecraft_central_body = "Jupiter"
body_settings.add_empty_settings(spacecraft_name)
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time_buffer, end_time_buffer, 10.0, spacecraft_central_body, global_frame_orientation)
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

# Ground Station Setup
receiving_station_name ='Yg'
new_ground_stations_settings = [environment_setup.ground_station.basic_station(
    receiving_station_name, [250, np.deg2rad(-29.0464), np.deg2rad(115.3456)], element_conversion.geodetic_position_type)]
body_settings.get('Earth').ground_station_settings = environment_setup.ground_station.radio_telescope_stations() + new_ground_stations_settings

bodies = environment_setup.create_system_of_bodies(body_settings)
body_fixed_station_position = bodies.get('Earth').get_ground_station(receiving_station_name).station_state.get_cartesian_position(0)

# Transponder and Frequencies
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("JUICE").system_models = vehicleSys

base_frequency = 8422.49e6
reception_band = observations_setup.ancillary_settings.FrequencyBands.x_band
transmission_band =  observations_setup.ancillary_settings.FrequencyBands.x_band

station_ramp = environment.PiecewiseLinearFrequencyInterpolator(
    [DateTime(2024,8,19, 10,58,36).epoch(), DateTime(2024,8,20, 11,32,17).epoch()],
    [DateTime(2024,8,19, 22,10,22).epoch(), DateTime(2024,8,20, 19,28,9).epoch()],
    [0, 0], [7180.142419e6, 7180.127320e6]
)
bodies.get('Earth').get_ground_station('NWNORCIA').transmitting_frequency_calculator = station_ramp

# %% [markdown]
# ## 4. Load (possibly Shifted by setting time_shift = x) FDETS Data and Compute Residuals

# %%
link_definition = observable_models_setup.links.LinkDefinition({
    observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'Yg'),
    observable_models_setup.links.reflector1: observable_models_setup.links.body_origin_link_end_id('JUICE'),
    observable_models_setup.links.transmitter: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
})

light_time_correction_list = list()
observable_models_setup.light_time_corrections.set_vmf_troposphere_data(
    ["juice_archive/vmf/2024231.vmf3_g", "juice_archive/vmf/2024232.vmf3_g",
     "juice_archive/vmf/2024233.vmf3_g", "juice_archive/vmf/2024234.vmf3_g"],
    True, False, bodies, True, True
)
light_time_correction_list.append(observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]))

# Load Shifted Observations
fdets_original = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/Fdets.jui2024.08.20.Yg.r2i.txt"

time_shift = 0.84
fdets_clean = strip_and_shift_first_column(fdets_original, "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA_shifted/", time_shift_seconds= time_shift)

column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max", "doppler_measured_frequency_hz", "doppler_noise_hz"]
station_positions = {'Yg': body_fixed_station_position}

print(f"Loading shifted FDETS observations ({time_shift} s)...")
fdets_collection = observations_setup.observations_wrapper.observations_from_fdets_files(
    fdets_clean, base_frequency, column_types, "JUICE", 'NWNORCIA', 'Yg',
    reception_band, transmission_band, station_positions
)

# Create Simulator and Compute Residuals
model_settings = [observable_models_setup.model_settings.doppler_measured_frequency(link_definition, light_time_correction_list)]
observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(model_settings, bodies)
estimation.observations.compute_residuals_and_dependent_variables(fdets_collection, observation_simulators, bodies)

# %% [markdown]
# ## 5. Visualization

# %%
times_tdb = fdets_collection.get_concatenated_observation_times()
observed_val = fdets_collection.get_concatenated_observations()
residuals = fdets_collection.get_concatenated_residuals()

from scipy import signal
residuals_detrended = signal.detrend(residuals)

simulated_val = observed_val - residuals_detrended

converter = time_representation.default_time_scale_converter()
times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(converter.convert_time(time_representation.tdb_scale, time_representation.utc_scale, t))) for t in times_tdb]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1.plot(times_datetime, observed_val, label=f'Actual ({time_shift}s Shift Applied)', color='blue', alpha=0.6)
ax1.plot(times_datetime, simulated_val, label='Simulated (Tudat)', color='red', linestyle='--')
ax1.set_ylabel('Frequency [Hz]'); ax1.set_title('JUICE Doppler Comparison (with 0.4s Data Shift)'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.scatter(times_datetime, residuals, color='gray', s=1, alpha=0.3, label='Original Residuals')
ax2.scatter(times_datetime, residuals_detrended, color='green', s=2, label='Detrended Residuals')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylabel('Residual [Hz]'); ax2.set_xlabel('Time (UTC)')
ax2.legend(title=f'Detrended RMS: {np.sqrt(np.mean(residuals_detrended**2)):.6f} Hz' + f'Original RMS: {np.sqrt(np.mean(residuals**2)):.6f} Hz'), ax2.grid(True, alpha=0.3)
ax2.legend(title=f'Original RMS: {np.sqrt(np.mean(residuals**2)):.6f} Hz'); ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate(); plt.tight_layout(); plt.show()