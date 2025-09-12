import numpy as np
import os

from tudatpy.astro import time_representation
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from grail_examples_functions import get_grail_files, get_grail_panel_geometry
import pickle
from collections import defaultdict
import random
from tudatpy.astro.time_representation import DateTime
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


import numpy as np
import os

from tudatpy.astro import time_representation
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from grail_examples_functions import get_grail_files, get_grail_panel_geometry
import pickle
from collections import defaultdict
import random
from tudatpy.astro.time_representation import DateTime
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_multi_arc_analysis_grail(base_path='parallel_case14_grail', num_arcs=1):
    """
    Plot spice, prefit, and postfit data from multiple arc pickle files.
    Creates separate figures for each data type, with legend showing RMS per link.
    """

    all_data = []

    # Load data from all arcs
    for idx in range(num_arcs):
        pickle_path = f'{base_path}/arc_{idx}/residDf.pkl'
        print(pickle_path)
        try:
            with open(pickle_path, 'rb') as pickle_file:
                arc_data = pickle.load(pickle_file)
                print(arc_data)
                arc_data['arc'] = idx
                all_data.append(arc_data)
                print(f"Loaded arc {idx}: {len(arc_data)} observations")
        except FileNotFoundError:
            print(f"Warning: File {pickle_path} not found, skipping arc {idx}")
        except Exception as e:
            print(f"Error loading arc {idx}: {e}")

    if not all_data:
        print("No data loaded! Check file paths.")
        return

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    # Filter out data points with residuals > 0.05 (only positive values)
    residual_threshold = 0.01
    original_length = len(df)
    df = df[(np.abs(df['postfit']) < residual_threshold)]
    print(df)
    print(f"Filtered data: {original_length- len(df)} observations with residuals > {residual_threshold}")

    print(f"\nTotal combined data: {len(df)} observations")
    print(f"Arcs loaded: {sorted(df['arc'].unique())}")
    print(f"Link ends: {df['link_ends'].unique()}")

    # Convert time to days from UTC0
    print(df)
    utc0_time = df['times'].min()
    time_days = (df['times'] - utc0_time) / (24 * 3600)

    doppler_times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in df['times']]

    # Extract downlink station from link_ends column
    df['downlink_station'] = df['link_ends'].str.split(' - ').str[1]

    # Get unique downlink stations and assign colors
    unique_downlinks = df['downlink_station'].unique()
    cmap = plt.cm.get_cmap('tab10', len(unique_downlinks))
    color_map = {downlink: cmap(i) for i, downlink in enumerate(unique_downlinks)}
    markers = ['o']

    # Plot each data type in a separate figure
    for data_col, ylabel, title in [
        ('prefit', 'Prefit (Hz)', 'Prefit vs Time (All Arcs)'),
        ('postfit', 'Postfit (Hz)', 'Postfit vs Time (All Arcs)')
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios':[4,1]})
        ax_main, ax_hist = axes

        # Get unique links for iteration but color by downlink
        unique_links = df['link_ends'].unique()

        for link in unique_links:
            mask = df['link_ends'] == link
            link_data = df[mask]

            # Get color based on downlink station for this link
            downlink_station = link.split(' - ')[1]
            color = color_map[downlink_station]

            unique_arcs = link_data['arc'].unique()
            for i, arc in enumerate(unique_arcs):
                arc_mask = link_data['arc'] == arc
                arc_link_data = link_data[arc_mask]
                arc_indices = mask & (df['arc'] == arc)
                alpha = 0.7 if len(unique_arcs) == 1 else 0.5
                ax_main.scatter(
                    time_days[arc_indices],
                    arc_link_data[data_col],
                    s=15, alpha=alpha,
                    color=color,
                    marker=markers[0],
                    edgecolors=color,
                    linewidths=0.2,
                )
            # Histogram for this link (colored by downlink)
            ax_hist.hist(
                link_data[data_col].values, bins=30,
                orientation='horizontal', alpha=0.6,
                color=color
            )

        # Format main plot
        ax_main.grid(True, linestyle='--', alpha=0.3)
        ax_main.set_ylabel('Residuals (Hz)', fontsize=15)
        ax_main.set_xlabel(f"Time [days from {doppler_times_datetime[0].strftime('%Y-%m-%d %H:%M:%S')}]", fontsize=15)
        # Format histogram
        ax_hist.set_xlabel("Count", fontsize=15)
        ax_hist.grid(True, linestyle='--', alpha=0.3)
        ax_hist.tick_params(left=False, labelleft=False)

        # --- Legend with RMS (grouped by downlink) ---
        legend_handles = []

        overall_rms = np.sqrt(np.mean(df[data_col].values**2))

        for downlink in unique_downlinks:
            # Get all data for this downlink station (regardless of uplink)
            downlink_data = df[df['downlink_station'] == downlink]
            rms_val = np.sqrt(np.mean(downlink_data[data_col].values**2))


            # Create legend handle with correct color and RMS
            if 'prefit' in data_col:
                handle = ax_main.scatter([], [], color=color_map[downlink],
                                         label=f"{downlink}, RMS: {rms_val:.2f} Hz")
                ax_main.set_title(f'Overall RMS: {overall_rms:.2f} Hz', fontsize=15)
            elif 'postfit' in data_col or 'spice' in data_col:
                ax_main.set_ylim(-0.015, 0.015)
                handle = ax_main.scatter([], [], color=color_map[downlink],
                                         label=f"{downlink}, RMS: {rms_val*1000:.2f} mHz")
                ax_main.set_title(f'Overall RMS: {overall_rms*1000:.2f} mHz', fontsize=15)
            legend_handles.append(handle)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        fig.legend(handles=legend_handles,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.43, 0.15),
                   fontsize=11)

        # === Save both PNG and PDF ===
        out_png = f"{base_path}/multi_arc_{data_col}.png"
        out_pdf = f"{base_path}/multi_arc_{data_col}.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png} and {out_pdf}")

def plot_multi_arc_analysis(base_path='parallel_case14', num_arcs=6):
    """
    Plot spice, prefit, and postfit data from multiple arc pickle files.
    Creates separate figures for each data type, with legend showing RMS per link.
    """

    all_data = []

    # Load data from all arcs
    for idx in range(num_arcs):
        pickle_path = f'{base_path}/arc_{idx}/residDf.pkl'
        try:
            with open(pickle_path, 'rb') as pickle_file:
                arc_data = pickle.load(pickle_file)
                arc_data['arc'] = idx
                all_data.append(arc_data)
                print(f"Loaded arc {idx}: {len(arc_data)} observations")
        except FileNotFoundError:
            print(f"Warning: File {pickle_path} not found, skipping arc {idx}")
        except Exception as e:
            print(f"Error loading arc {idx}: {e}")

    if not all_data:
        print("No data loaded! Check file paths.")
        return

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined data: {len(df)} observations")
    print(f"Arcs loaded: {sorted(df['arc'].unique())}")
    print(f"Link ends: {df['link_ends'].unique()}")

    # Convert time to days from UTC0
    utc0_time = df['time'].min()
    time_days = (df['time'] - utc0_time) / (24 * 3600)

    doppler_times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in df['time']]

    # Extract downlink station from link_ends column
    df['downlink_station'] = df['link_ends'].str.split(' - ').str[1]

    # Get unique downlink stations and assign colors
    unique_downlinks = df['downlink_station'].unique()
    cmap = plt.cm.get_cmap('tab10', len(unique_downlinks))
    color_map = {downlink: cmap(i) for i, downlink in enumerate(unique_downlinks)}
    markers = ['o']

    # Plot each data type in a separate figure
    for data_col, ylabel, title in [
        ('spice', 'Spice', 'Spice vs Time (All Arcs)'),
        ('prefit', 'Prefit (Hz)', 'Prefit vs Time (All Arcs)'),
        ('postfit', 'Postfit (Hz)', 'Postfit vs Time (All Arcs)')
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios':[4,1]})
        ax_main, ax_hist = axes

        # Get unique links for iteration but color by downlink
        unique_links = df['link_ends'].unique()

        for link in unique_links:
            mask = df['link_ends'] == link
            link_data = df[mask]

            # Get color based on downlink station for this link
            downlink_station = link.split(' - ')[1]
            color = color_map[downlink_station]

            unique_arcs = link_data['arc'].unique()
            for i, arc in enumerate(unique_arcs):
                arc_mask = link_data['arc'] == arc
                arc_link_data = link_data[arc_mask]
                arc_indices = mask & (df['arc'] == arc)
                alpha = 0.7 if len(unique_arcs) == 1 else 0.5
                ax_main.scatter(
                    time_days[arc_indices],
                    arc_link_data[data_col],
                    s=15, alpha=alpha,
                    color=color,
                    marker=markers[0],
                    edgecolors=color,
                    linewidths=0.2,
                )
            # Histogram for this link (colored by downlink)
            ax_hist.hist(
                link_data[data_col].values, bins=30,
                orientation='horizontal', alpha=0.6,
                color=color
            )

        # Format main plot
        ax_main.grid(True, linestyle='--', alpha=0.3)
        ax_main.set_ylabel('Residuals (Hz)', fontsize=15)
        ax_main.set_xlabel(f"Time [days from {doppler_times_datetime[0].strftime('%Y-%m-%d %H:%M:%S')}]", fontsize=15)

        # Format histogram
        ax_hist.set_xlabel("Count", fontsize=15)
        ax_hist.grid(True, linestyle='--', alpha=0.3)
        ax_hist.tick_params(left=False, labelleft=False)

        # --- Legend with RMS (grouped by downlink) ---
        legend_handles = []

        overall_rms = np.sqrt(np.mean(df[data_col].values**2))

        for downlink in unique_downlinks:
            # Get all data for this downlink station (regardless of uplink)
            downlink_data = df[df['downlink_station'] == downlink]
            rms_val = np.sqrt(np.mean(downlink_data[data_col].values**2))


            # Create legend handle with correct color and RMS
            if 'prefit' in data_col:
                handle = ax_main.scatter([], [], color=color_map[downlink],
                                         label=f"{downlink}, RMS: {rms_val:.2f} Hz")
                ax_main.set_title(f'Overall RMS: {overall_rms:.2f} Hz', fontsize=15)
            elif 'postfit' in data_col or 'spice' in data_col:
                ax_main.set_ylim(-0.10, 0.09)
                handle = ax_main.scatter([], [], color=color_map[downlink],
                                         label=f"{downlink}, RMS: {rms_val*1000:.2f} mHz")
                ax_main.set_title(f'Overall RMS: {overall_rms*1000:.2f} mHz', fontsize=15)
            legend_handles.append(handle)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        fig.legend(handles=legend_handles,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.43, 0.15),
                   fontsize=11)

        # === Save both PNG and PDF ===
        out_png = f"{base_path}/multi_arc_{data_col}.png"
        out_pdf = f"{base_path}/multi_arc_{data_col}.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png} and {out_pdf}")

        plt.show()
def plot_mro_pre_post_fit_orbit(prefit=True, postfit=False):
    base_path = 'parallel_case14'

    # Only keep entries that are directories
    arc_dirs = [
        arc_dir for arc_dir in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, arc_dir))
    ]

    all_prefit_diff = []
    all_postfit_diff = []

    plt.rcParams.update({"font.size": 15})

    print(arc_dirs)
    for arc_dir in arc_dirs:
        full_path = os.path.join(base_path, arc_dir)
        print(full_path)

        arc_index = os.path.basename(arc_dir).split("_")[1]
        print(f"Processing state differences from {full_path}")

        if int(arc_index) > 4:
            continue

        try:
            # Load prefit state differences
            if prefit:
                prefit_df = pd.read_pickle(f"{base_path}/{arc_dir}/rsw_state_difference_prefit.pkl")
                prefit_df["arc_index"] = arc_index
                all_prefit_diff.append((arc_index, prefit_df))

            # Load postfit state differences
            if postfit:
                postfit_df = pd.read_pickle(f"{base_path}/{arc_dir}/rsw_state_difference_postfit.pkl")
                postfit_df["arc_index"] = arc_index
                all_postfit_diff.append((arc_index, postfit_df))
        except FileNotFoundError as e:
            print(f"Warning: Could not load state difference data from {arc_dir}: {e}")

    # === Compute overall RMS across all arcs ===
    overall_rms = {}
    components = ["R", "T", "N"]

    if postfit and all_postfit_diff:
        all_postfit_dfs = [df for _, df in all_postfit_diff]
        combined_postfit_df = pd.concat(all_postfit_dfs, ignore_index=True)
        for component in components:
            overall_rms[component] = np.sqrt(
                np.mean(np.square(combined_postfit_df[component]))
            )

        print("\nOverall RMS values across all arcs (postfit):")
        for c in components:
            print(f"{c} component: {overall_rms[c]:.2f} m")

        pos_rms_3d = np.sqrt(sum(overall_rms[c]**2 for c in components))
        print(f"3D position RMS: {pos_rms_3d:.2f} m")

    # === Plotting ===
    if (prefit and all_prefit_diff) or (postfit and all_postfit_diff):
        # Sort by arc index
        if prefit and all_prefit_diff:
            all_prefit_diff.sort(key=lambda x: int(x[0]))
        if postfit and all_postfit_diff:
            all_postfit_diff.sort(key=lambda x: int(x[0]))

        # Find global min time for consistent x-axis
        if prefit and all_prefit_diff:
            global_t_min = min([df["t"].min() for _, df in all_prefit_diff])
        elif postfit and all_postfit_diff:
            global_t_min = min([df["t"].min() for _, df in all_postfit_diff])

        # Determine number of rows based on flags
        n_rows = sum([prefit, postfit])

        # Create figure with appropriate grid
        if n_rows == 1:
            fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
            axes = axes.reshape(1, -1)  # ensure 2D for indexing
        else:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)

        component_colors = {"R": "tab:blue", "T": "tab:orange", "N": "tab:green"}
        current_row = 0

        # Plot prefit
        if prefit and all_prefit_diff:
            for i, (arc_index, df) in enumerate(all_prefit_diff):
                time_days = (df["t"] - global_t_min) / 86400
                for col, component in enumerate(components):
                    rms_label = f"{component}, RMS: {overall_rms.get(component, np.nan):.2f} m"
                    axes[current_row, col].scatter(
                        time_days,
                        df[component],
                        label=rms_label if i == 0 else "",
                        edgecolor = component_colors[component],
                        color=component_colors[component],
                        alpha=0.5,
                        s=3,
                    )
                    if i < len(all_prefit_diff) - 1:
                        axes[current_row, col].axvline(
                            time_days.max(), color="gray", linestyle="-", linewidth=0.5, alpha=0.3
                        )
            current_row += 1

        # Plot postfit
        if postfit and all_postfit_diff:
            for i, (arc_index, df) in enumerate(all_postfit_diff):
                time_days = (df["t"] - global_t_min) / 86400
                for col, component in enumerate(components):
                    rms_label = f"{component}, RMS: {overall_rms.get(component, np.nan):.2f} m"
                    axes[current_row, col].scatter(
                        time_days,
                        df[component],
                        label=rms_label if i == 0 else "",
                        edgecolor = component_colors[component],
                        color=component_colors[component],
                        alpha=0.5,
                        s=3,
                    )
                    if i < len(all_postfit_diff) - 1:
                        axes[current_row, col].axvline(
                            time_days.max(), color="gray", linestyle="-", linewidth=0.5, alpha=0.3
                        )

        # Set titles and labels
        for row in range(n_rows):
            for col, component in enumerate(components):
                axes[row, col].set_ylabel(f"Difference [m]")
                axes[row, col].legend(loc = 'upper center')
                axes[row, col].grid(which="both", linestyle="--", linewidth=1.5)

        for col in range(3):
            axes[n_rows-1, col].set_xlabel("Time [days]")

        # Save
        if prefit and postfit:
            suffix = "prefit_and_postfit"
        elif prefit:
            suffix = "prefit_only"
        else:
            suffix = "postfit_only"

        plt.tight_layout()
        plt.savefig(
            f"{base_path}/combined_state_differences_{suffix}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved combined state differences plot to {base_path}/combined_state_differences_{suffix}.png")

def plot_mro_doppler_and_range(range_flag = False, doppler_flag = False):
    nb_cores = 4
    time_scale_converter = time_representation.default_time_scale_converter()

    if doppler_flag:

        # Load Doppler data
        doppler_times, doppler_residuals = [], []
        all_link_ids_doppler = []
        for i in range(nb_cores):
            with open(f'outputs_mro/mro_doppler_link_end_ids_{i}_names.pkl', 'rb') as pickle_file:
                all_link_ids_doppler.append(pickle.load(pickle_file))
            doppler_times.append(np.loadtxt(f"outputs_mro/mro_doppler_filtered_time_{i}.dat", delimiter=","))
            doppler_residuals.append(np.loadtxt(f"outputs_mro/mro_doppler_filtered_residuals_{i}.dat", delimiter=","))

        doppler_times = np.concatenate(doppler_times)

        # Convert time to UTC datetime
        doppler_times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in doppler_times]

        doppler_times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in doppler_times_utc]

        doppler_residuals = np.concatenate(doppler_residuals)
        link_ends_ids_doppler = np.concatenate(all_link_ids_doppler)

        # Identify unique receivers and assign colors
        unique_receivers = sorted({link_id.split(" - ")[1] for link_id in link_ends_ids_doppler})
        colormap = plt.cm.get_cmap('tab10', len(unique_receivers))
        receiver_colors = {receiver: colormap(i) for i, receiver in enumerate(unique_receivers)}

        # Dictionary to collect RMS values for each unique receiver
        station_rms_values = defaultdict(list)

        # === Doppler Plot ===
        fig_doppler = plt.figure(figsize=(12, 5))
        gs_doppler = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.027)
        ax_doppler = fig_doppler.add_subplot(gs_doppler[0])
        ax_hist_doppler = fig_doppler.add_subplot(gs_doppler[1], sharey=ax_doppler)

        added_labels = set()

        overall_rms_doppler = np.sqrt(np.mean(doppler_residuals**2))

        for link_id in np.unique(link_ends_ids_doppler):
            mask = (link_ends_ids_doppler == link_id)
            transmitter, receiver = link_id.split(" - ")

            # Calculate RMS for this station
            station_residuals = doppler_residuals[mask]
            rms_value = np.sqrt(np.mean(station_residuals**2))
            station_rms_values[receiver].append(rms_value)

            marker = 'o' if transmitter == receiver else 'D'
            color = receiver_colors[receiver]
            label = receiver if receiver not in added_labels else None

            ax_doppler.scatter(
                np.array(doppler_times_datetime)[mask],
                doppler_residuals[mask],
                s=5, color=color, label=label, marker=marker, alpha=0.4,
                facecolors='none', edgecolors=color,
            )

            ax_hist_doppler.hist(
                doppler_residuals[mask], bins=40,
                orientation='horizontal', alpha=0.5,
                color=color, label=label
            )

            added_labels.add(receiver)

        # Format Doppler plot
        ax_doppler.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_doppler.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig_doppler.autofmt_xdate()
        ax_doppler.set_title(f'Overall RMS: {overall_rms_doppler*1000:.2f} mHz', fontsize = 15)
        ax_doppler.set_xlabel("UTC Date", fontsize=15)
        ax_doppler.set_ylabel("Doppler Residuals (Hz)", fontsize=15)
        ax_doppler.grid(True, linestyle="--", alpha=0.3)
        ax_hist_doppler.set_xlabel("Count")
        ax_hist_doppler.grid(True, linestyle="--", alpha=0.3)
        ax_hist_doppler.tick_params(left=False)
        plt.setp(ax_hist_doppler.get_yticklabels(), visible=False)
        ax_doppler.set_ylim([-0.1, 0.1])

        # Update legend with RMS values
        avg_rms_text = {station: np.mean(values) for station, values in station_rms_values.items()}
        handles, labels = ax_doppler.get_legend_handles_labels()
        new_labels = [f"{station}, RMS: {avg_rms_text[station]*1000:.2f}" if station in avg_rms_text else station
                      for station in labels]
        ax_doppler.legend(handles, new_labels, ncols=3, loc='lower center', fontsize=10)

        plt.tight_layout()

        # === Save both PNG and PDF ===
        out_png = "mro_doppler_residuals.png"
        out_pdf = "mro_doppler_residuals.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png} and {out_pdf}")

        plt.show()


    if range_flag:
        # === Range Plot ===
        fig_range = plt.figure(figsize=(12, 5))
        gs_range = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.027)

        ax_range = fig_range.add_subplot(gs_range[0])
        ax_hist_range = fig_range.add_subplot(gs_range[1], sharey=ax_range)

        range_times, range_residuals, range_conversion_factors, all_link_ids = [], [], [], []
        for i in range(nb_cores):
            with open(f'outputs_mro/mro_range_link_end_ids_{i}_names.pkl', 'rb') as pickle_file:
                all_link_ids.append(pickle.load(pickle_file))
            range_times.append(np.loadtxt(f"outputs_mro/mro_range_filtered_time_{i}.dat", delimiter=","))
            range_residuals.append(np.loadtxt(f"outputs_mro/mro_range_filtered_residuals_{i}.dat", delimiter=","))
            range_conversion_factors.append(np.loadtxt(f"outputs_mro/mro_range_conversion_factors_{i}.dat", delimiter=","))

        range_times = np.concatenate(range_times)

        # Convert to UTC datetime
        range_times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in range_times]

        range_times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in range_times_utc]

        range_residuals = np.concatenate(range_residuals)
        range_conversion_factors = np.concatenate(range_conversion_factors)
        link_ends_ids = np.concatenate(all_link_ids)
        unique_ids = np.unique(link_ends_ids)
        colors = plt.cm.get_cmap('tab10', len(unique_ids))

        # Dictionary to collect RMS values for each unique receiver
        station_rms_values = defaultdict(list)

        filtered_residuals_scaled = np.array([range_residuals *  range_conversion_factors])
        print(filtered_residuals_scaled)

        overall_rms_range = np.sqrt(np.mean(filtered_residuals_scaled**2))

        for j, link_id in enumerate(unique_ids):
            mask = (link_ends_ids == link_id)
            transmitter_name, receiver_name = link_id.split(" - ")
            residuals_scaled = (range_residuals * range_conversion_factors)[mask]

            # Calculate RMS for this specific pass and store it
            rms_value = np.sqrt(np.mean(residuals_scaled**2))
            station_rms_values[receiver_name].append(rms_value)

            print(receiver_name)
            if transmitter_name != receiver_name:
                ax_range.scatter(np.array(range_times_datetime)[mask], residuals_scaled,
                                 s=10, color=colors(j), label=receiver_name, marker='D', facecolors='none', edgecolors=colors(j))
                ax_hist_range.hist(residuals_scaled, bins=40, orientation='horizontal',
                                   alpha=0.4, color=colors(j), label=receiver_name)
            else:
                ax_range.scatter(np.array(range_times_datetime)[mask], residuals_scaled,
                                 s=10, color=colors(j), label=receiver_name, marker='o', facecolors='none', edgecolors=colors(j))
                ax_hist_range.hist(residuals_scaled, bins=40, orientation='horizontal',
                                   alpha=0.4, color=colors(j), label=receiver_name)

        # Calculate average RMS for each unique station
        rms_text = []
        for station, rms_values in station_rms_values.items():
            avg_rms = np.mean(rms_values)
            rms_text.append(f"{station}: {avg_rms:.2f} m")

        ax_range.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_range.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig_range.autofmt_xdate()
        ax_range.set_title(f'Overall RMS: {overall_rms_range:.2f} m', fontsize = 15)
        ax_range.set_xlabel("UTC Date", fontsize=15)
        ax_range.set_ylabel("Residuals [m]", fontsize=15)
        ax_range.grid(True, linestyle="--", alpha=0.3)
        ax_range.legend(ncols=3, loc = 'lower center', fontsize = 11)
        ax_hist_range.set_xlabel("Count")
        ax_hist_range.grid(True, linestyle="--", alpha=0.3)
        ax_hist_range.tick_params(left=False)
        plt.setp(ax_hist_range.get_yticklabels(), visible=False)

        avg_rms_text = {station: np.mean(values) for station, values in station_rms_values.items()}
        handles, labels = ax_range.get_legend_handles_labels()
        new_labels = [f"{label}, RMS: {avg_rms_text[label]:.2f} m" if label in avg_rms_text else label
                      for label in labels]
        ax_range.legend(handles, new_labels, ncols=3, loc='lower center', fontsize = 10)
        # === Save both PNG and PDF ===
        out_png = "mro_range_residuals.png"
        out_pdf = "mro_range_residuals.pdf"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_png} and {out_pdf}")

        plt.show()
###############
def plot_combined_grail_doppler_pre_fit():
    nb_cores = 4
    time_scale_converter = time_representation.default_time_scale_converter()

    all_times = []
    all_residuals = []
    all_link_ids = []

    # Collect data from all cores
    for core_idx in range(nb_cores):
        with open(f'Archive_grail/grail_residuals_output/link_end_ids_{core_idx}_names.pkl', 'rb') as pickle_file:
            link_ids = pickle.load(pickle_file)

        times = np.loadtxt(f"Archive_grail/grail_residuals_output/observation_times_{core_idx}.dat", delimiter=",")
        residuals = np.loadtxt(f"Archive_grail/grail_residuals_output/residuals_wrt_spice_{core_idx}.dat", delimiter=",")

        times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in times]
        times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in times_utc]

        all_times.extend(times_datetime)
        all_residuals.extend(residuals)
        all_link_ids.extend(link_ids)

    # Convert to NumPy arrays
    all_times = np.array(all_times)
    all_residuals = np.array(all_residuals)

    overall_rms = np.sqrt(np.mean(all_residuals**2))

    all_link_ids = np.array(all_link_ids)

    unique_links = np.unique(all_link_ids)
    cmap = plt.cm.get_cmap('tab10', len(unique_links))
    color_map = {link_id: cmap(i) for i, link_id in enumerate(unique_links)}

    # Create figure with shared axes
    fig, (ax_doppler, ax_hist) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [4, 1]})

    rms_text = []

    for link_id in unique_links:
        mask = all_link_ids == link_id
        transmitter, receiver = link_id.split(" - ")
        marker = 'o' if transmitter == receiver else 'D'
        color = color_map[link_id]

        # Calculate RMS for this station
        station_residuals = all_residuals[mask]
        rms_value = np.sqrt(np.mean(station_residuals**2))
        rms_text.append(f"{receiver}: {rms_value:.4f} Hz")

        ax_doppler.scatter(
            all_times[mask],
            all_residuals[mask],
            s=10, alpha=0.6,
            facecolors='none', edgecolors=color,
            label=f"{receiver}, RMS: {rms_value*1000:.2f} mHz", marker=marker
        )
        ax_hist.hist(
            all_residuals[mask], bins=40,
            orientation='horizontal', alpha=0.4,
            color=color, label=receiver
        )

    # Format axes
    ax_doppler.set_title(f'Overall RMS: {overall_rms*1000:.2f} mHz', fontsize = 15)
    ax_doppler.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_doppler.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax_doppler.grid(True, linestyle='--', alpha=0.3)
    ax_doppler.set_ylabel("Doppler Residuals (Hz)", fontsize=15)
    ax_doppler.set_xlabel("UTC Date", fontsize=15)

    ax_hist.set_xlabel("Count", fontsize=15)
    ax_hist.grid(True, linestyle='--', alpha=0.3)
    ax_hist.tick_params(left=False)
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    # Unified legend
    ax_doppler.legend(loc='lower center', ncols=3, fontsize = 11)

    plt.tight_layout()
    out_png = "grail_prefit_residuals.png"
    out_pdf = "grail_prefit_residuals.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png} and {out_pdf}")

    plt.show()

def plot_grail_doppler_pre_fit():
    nb_cores = 4

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(nb_cores, 2, width_ratios=[4, 1], wspace=0.027)

    time_scale_converter = time_representation.default_time_scale_converter()

    for core_idx in range(nb_cores):
        # === Load Doppler Data ===
        with open(f'Archive_grail/grail_residuals_output/link_end_ids_{core_idx}_names.pkl', 'rb') as pickle_file:
            link_ids = pickle.load(pickle_file)

        times = np.loadtxt(f"Archive_grail/grail_residuals_output/observation_times_{core_idx}.dat", delimiter=",")
        residuals = np.loadtxt(f"Archive_grail/grail_residuals_output/filtered_residuals_wrt_spice_{core_idx}.dat", delimiter=",")

        times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in times]
        times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in times_utc]

        link_ids = np.array(link_ids)
        residuals = np.array(residuals)
        times_datetime = np.array(times_datetime)

        unique_links = np.unique(link_ids)
        cmap = plt.cm.get_cmap('tab10', len(unique_links))
        color_map = {link_id: cmap(i) for i, link_id in enumerate(unique_links)}

        # === Subplots ===
        ax_doppler = fig.add_subplot(gs[core_idx, 0])
        ax_hist = fig.add_subplot(gs[core_idx, 1], sharey=ax_doppler)

        rms_text = []

        for j, link_id in enumerate(unique_links):
            mask = link_ids == link_id
            transmitter, receiver = link_id.split(" - ")
            marker = 'o' if transmitter == receiver else 'D'
            color = color_map[link_id]

            # Calculate RMS for this station
            station_residuals = residuals[mask]
            rms_value = np.sqrt(np.mean(station_residuals**2))
            rms_text.append(f"{receiver}: {rms_value:.4f} Hz")

            ax_doppler.scatter(
                times_datetime[mask],
                residuals[mask],
                s=5, alpha=0.6,
                facecolors='none', edgecolors=color,
                label=link_id.split(' - ')[1], marker=marker
            )
            ax_hist.hist(
                residuals[mask], bins=40,
                orientation='horizontal', alpha=0.4,
                color=color, label=link_id.split(' - ')[1]
            )

        # Format axes
        ax_doppler.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_doppler.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax_doppler.grid(True, linestyle='--', alpha=0.3)
        ax_doppler.set_ylabel("Doppler Residuals (Hz)", fontsize=15)
        if core_idx == nb_cores - 1:
            ax_doppler.set_xlabel("UTC Date", fontsize=15)

        ax_doppler.set_ylim(-0.1, 0.1)
        ax_hist.set_xlabel("Count", fontsize=15)
        ax_hist.grid(True, linestyle='--', alpha=0.3)
        ax_hist.tick_params(left=False)
        plt.setp(ax_hist.get_yticklabels(), visible=False)

        # Add legend per subplot
        ax_doppler.legend(loc='lower center', ncols=4, fontsize = 11)

        # Add RMS text box
        rms_text_str = "RMS:\n" + "\n".join(rms_text)
        ax_doppler.text(0.02, 0.98, rms_text_str, transform=ax_doppler.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()



def plot_grail_doppler_post_fit():
    nb_cores = 4

    fig = plt.figure(figsize=(12, 5 * nb_cores))
    gs = gridspec.GridSpec(nb_cores, 2, width_ratios=[4, 1], wspace=0.027)

    time_scale_converter = time_representation.default_time_scale_converter()

    for core_idx in range(nb_cores):
        # === Load Doppler Data ===
        with open(f'Archive_grail/grail_estimation_output/link_end_ids_{core_idx}_names.pkl', 'rb') as pickle_file:
            link_ids = pickle.load(pickle_file)

        times = np.loadtxt(f"Archive_grail/grail_estimation_output/observation_times_{core_idx}.dat", delimiter=",")
        residuals = np.loadtxt(f"Archive_grail/grail_estimation_output/postfit_residuals_{core_idx}.dat", delimiter=",")

        times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in times]
        times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in times_utc]

        link_ids = np.array(link_ids)
        residuals = np.array(residuals)
        times_datetime = np.array(times_datetime)

        unique_links = np.unique(link_ids)
        cmap = plt.cm.get_cmap('tab10', len(unique_links))
        color_map = {link_id: cmap(i) for i, link_id in enumerate(unique_links)}

        # === Subplots ===
        ax_doppler = fig.add_subplot(gs[core_idx, 0])
        ax_hist = fig.add_subplot(gs[core_idx, 1], sharey=ax_doppler)

        rms_text = []

        for j, link_id in enumerate(unique_links):
            mask = link_ids == link_id
            transmitter, receiver = link_id.split(" - ")
            marker = 'o' if transmitter == receiver else 'D'
            color = color_map[link_id]

            # Calculate RMS for this station
            station_residuals = residuals[mask]
            rms_value = np.sqrt(np.mean(station_residuals**2))
            rms_text.append(f"{receiver}: {rms_value:.2f} Hz")

            ax_doppler.scatter(
                times_datetime[mask],
                residuals[mask],
                s=5, alpha=0.6,
                facecolors='none', edgecolors=color,
                label=link_id.split(' - ')[1], marker=marker
            )
            ax_hist.hist(
                residuals[mask], bins=40,
                orientation='horizontal', alpha=0.4,
                color=color, label=link_id.split(' - ')[1]
            )

        # Format axes
        ax_doppler.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_doppler.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax_doppler.grid(True, linestyle='--', alpha=0.3)
        ax_doppler.set_ylabel("Doppler Residuals (Hz)", fontsize=15)
        if core_idx == nb_cores - 1:
            ax_doppler.set_xlabel("UTC Date", fontsize=15)

        ax_hist.set_xlabel("Count", fontsize=15)
        ax_hist.grid(True, linestyle='--', alpha=0.3)
        ax_hist.tick_params(left=False)
        plt.setp(ax_hist.get_yticklabels(), visible=False)

        # Add legend per subplot
        ax_doppler.legend(loc='lower center', ncols=4, fontsize = 11)

    plt.tight_layout()

    plt.tight_layout()
    out_png = "grail_postfit_residuals.png"
    out_pdf = "grail_postfit_residuals.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png} and {out_pdf}")

    plt.show()

def plot_combined_grail_doppler_post_fit():
    nb_cores = 6
    time_scale_converter = time_representation.default_time_scale_converter()

    all_times = []
    all_residuals = []
    all_link_ids = []

    # Collect data from all cores
    for core_idx in range(nb_cores):
        # === Load Doppler Data ===
        with open(f'Archive_grail/grail_estimation_output/link_end_ids_{core_idx}_names.pkl', 'rb') as pickle_file:
            link_ids = pickle.load(pickle_file)

        times = np.loadtxt(f"Archive_grail/grail_estimation_output/observation_times_{core_idx}.dat", delimiter=",")
        residuals = np.loadtxt(f"Archive_grail/grail_estimation_output/postfit_residuals_{core_idx}.dat", delimiter=",")

        times_utc = [time_scale_converter.convert_time(
            input_scale=time_representation.tdb_scale,
            output_scale=time_representation.utc_scale,
            input_value=t) for t in times]
        times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t)) for t in times_utc]

        all_times.extend(times_datetime)
        all_residuals.extend(residuals)
        all_link_ids.extend(link_ids)

    # Convert to NumPy arrays
    all_times = np.array(all_times)
    all_residuals = np.array(all_residuals)
    all_link_ids = np.array(all_link_ids)

    overall_rms = np.sqrt(np.mean(all_residuals**2))

    unique_links = np.unique(all_link_ids)
    cmap = plt.cm.get_cmap('tab10', len(unique_links))
    color_map = {link_id: cmap(i) for i, link_id in enumerate(unique_links)}

    # Create figure with shared axes
    fig, (ax_doppler, ax_hist) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [4, 1]})

    rms_text = []

    for link_id in unique_links:
        mask = all_link_ids == link_id
        transmitter, receiver = link_id.split(" - ")
        marker = 'o' if transmitter == receiver else 'D'
        color = color_map[link_id]

        # Calculate RMS for this station
        station_residuals = all_residuals[mask]
        rms_value = np.sqrt(np.mean(station_residuals**2))
        rms_text.append(f"{receiver}: {rms_value:.4f} Hz")

        print(np.std(all_residuals))
        ax_doppler.scatter(
            all_times[mask],
            all_residuals[mask],
            s=10, alpha=0.6,
            facecolors='none', edgecolors=color,
            label=f"{receiver}, RMS: {rms_value*1000:.2f} mHz", marker=marker
        )
        ax_hist.hist(
            all_residuals[mask], bins=40,
            orientation='horizontal', alpha=0.4,
            color=color, label=receiver
        )

    # Format axes
    ax_doppler.set_title(f"Overall RMS: {overall_rms*1000:.2f} mHz")
    ax_doppler.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_doppler.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax_doppler.grid(True, linestyle='--', alpha=0.3)
    ax_doppler.set_ylabel("Doppler Residuals (Hz)", fontsize=15)
    ax_doppler.set_xlabel("UTC Date", fontsize=15)

    ax_hist.set_xlabel("Count", fontsize=15)
    ax_hist.grid(True, linestyle='--', alpha=0.3)
    ax_hist.tick_params(left=False)
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    # Unified legend
    ax_doppler.legend(loc='lower center', ncols=3, fontsize = 11)

    plt.tight_layout()
    out_png = "grail_postfit_residuals.png"
    out_pdf = "grail_postfit_residuals.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png} and {out_pdf}")

    plt.show()

def plot_combined_grail_rsw_errors():

    base_folder = "Archive_grail/grail_estimation_output"
    nb_setups = 6  # number of cores / arcs

    colors = ['royalblue', 'orange', 'green']
    components = ['Radial', 'Along-Track', 'Cross-Track']

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    time_scale_converter = time_representation.default_time_scale_converter()

    # Collect all data for RMS calculation
    all_residuals = [[] for _ in range(3)]  # For each component

    # Variables to track global time reference
    global_t0 = None
    t0_datetime = None
    cumulative_time_offset = 0
    all_times_plot = []
    all_residuals_plot = [[] for _ in range(3)]

    for setup_idx in range(nb_setups+1):
        file_path = os.path.join(
            base_folder,
            f"postfit_rsw_state_difference_{setup_idx}.dat"
        )

        try:
            data = np.loadtxt(file_path, delimiter=',')
            times = data[:, 0]

            times_utc = [time_scale_converter.convert_time(
                input_scale=time_representation.tdb_scale,
                output_scale=time_representation.utc_scale,
                input_value=t) for t in times]

            # Set global reference time from first arc
            if global_t0 is None:
                global_t0 = times_utc[0]
                t0_datetime = DateTime.to_python_datetime(DateTime.from_epoch(global_t0))

            # Convert times to days from global start time
            times_days = [(t - global_t0) / 86400.0 for t in times_utc]  # Convert seconds to days

            # Add cumulative offset for subsequent arcs
            #times_days_offset = [t + cumulative_time_offset for t in times_days]

            residuals = data[:, 1:4]

            # Collect data for RMS calculation
            for comp_idx in range(3):
                all_residuals[comp_idx].extend(residuals[:, comp_idx])
                all_residuals_plot[comp_idx].extend(residuals[:, comp_idx])

            #all_times_plot.extend(times_days_offset)

            # Plot each component
            for comp_idx in range(3):
                ax.plot(
                    times_days,
                    residuals[:, comp_idx],
                    color=colors[comp_idx],
                    linewidth=1,
                    label=components[comp_idx] if setup_idx == 0 else ""  # only label once
                )

            # Update cumulative offset for next arc
            # Add a small gap (e.g., 0.1 days) between arcs for visual separation
            #if times_days_offset:
            #    cumulative_time_offset = max(times_days_offset) + 0.1

        except OSError:
            print(f"File not found: {file_path}")
            continue

    # Calculate RMS for each component and create legend
    new_labels = []
    for comp_idx in range(3):
        if all_residuals[comp_idx]:  # Check if data exists
            rms_value = np.sqrt(np.mean(np.array(all_residuals[comp_idx])**2))
            new_label = f"{components[comp_idx]}, RMS: {rms_value:.2f} m"
            new_labels.append(new_label)

    # Update legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:3], new_labels, ncols=3, loc='lower right', fontsize=11)

    ax.grid(linestyle='--', alpha=0.3)
    ax.set_xlabel(f"Time [days from {t0_datetime.strftime('%Y-%m-%d %H:%M:%S')}]", fontsize=15)
    ax.set_ylabel("Residuals [m]", fontsize=15)

    plt.tight_layout()
    out_png = "grail_orbit_postfit_residuals.png"
    out_pdf = "grail_orbit_postfit_residuals.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png} and {out_pdf}")
    plt.show()

#plot_combined_mro_doppler_post_fit()
#plot_combined_grail_rsw_errors()
#plot_combined_grail_doppler_pre_fit()
#plot_combined_grail_doppler_post_fit()
#plot_combined_grail_rsw_errors()

#plot_mro_pre_post_fit_orbit()
plot_multi_arc_analysis_grail()

#plot_mro_doppler_and_range(range_flag=True)
