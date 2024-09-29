# Tudatpy examples

Welcome to the repository showcasing example applications set up with Tudatpy!

If you want to know more about Tudatpy, please visit the [Tudat website](https://docs.tudat.space/en/latest/).
The website also holds the [examples rendered as notebooks](https://docs.tudat.space/en/latest/_src_getting_started/examples.html).
Any update to the examples in this repository will automatically update the [website repository](https://github.com/tudat-team/tudat-space) via the [Sync tudat-space submodule](https://github.com/tudat-team/tudatpy-examples/actions/workflows/sync-tudat-space.yml) action.

## Content

The examples are organized in different categories.

### Estimation

Examples related to state estimation.

- ``covariance_estimated_parameters``: setup of an orbit estimation problem, definition and propagation of the covariance matrix.
- ``estimation_dynamical_models``: application of different dynamical models to the simulation of observations and the estimation.
- ``full_estimation_example``: full estimation of individual parameters.
- ``retrieving_mpc_observation_data``: using Tudat's `BatchMPC` class for the retrieval and processing of observational data of minor planets, comets and outer irregular natural satellites of the major planets.
- ``estimation_with_mpc``: using real observational data from the Minor Planet Center (MPC) for the initial state estimation of a minor body.
- ``improved_estimation_with_mpc``: extension of the ``estimation_with_mpc`` example. Introduce and compare the effects of including satellite data, star catalog corrections, observation weighting and more expansive acceleration models in the estimation, retrieval of JPL Horizons data.
- ``galilean_moons_state_estimation``: using ephemeris data to simulate observations and enhance the accuracy of predicted orbits of the Galilean moons.
- ``mro_range_estimation``: loading tracking observations from Mars Reconnaissance Orbiter (MRO) with a variety of Deep Space Network (DSN) ground stations.

### Mission Design

Examples related to mission design.

- ``cassini1_mga_optimization``: using PyGMO to optimize an interplanetary transfer trajectory simulated using the multiple gravity assist (MGA) module of Tudat.
- ``hodographic_shaping_mga_optimization``: extension of the ``cassini1_mga_optimization`` example. Optimization of a low-thrust interplanetary transfer trajectory using the hodographic shaping method for the low-thrust legs.
- ``earth_mars_transfer_window``: usage of the Tudatpy's `porkchop` module to determine an optimal launch window (departure and arrival date) for an Earth-Mars transfer mission.
- ``low_thrust_earth_mars_transfer_window``: extension of the ``earth_mars_transfer_window`` example, modelling the interplanetary leg as low-thrust leg.

### Propagation

Examples related to state propagation.

Introductory examples:

- ``keplerian_satellite_orbit``: simulation of a Keplerian orbit around Earth (two-body problem).
- ``perturbed_satellite_orbit``: simulation of a perturbed orbit around Earth.
- ``linear_sensitivity_analysis``: extension of the ``perturbed_satellite_orbit`` example to propagate variational equations to perform a sensitivity analysis.
- ``solar_system_propagation``: numerical propagation of solar-system bodies, showing how a hierarchical, multi-body simulation can be set up.
- ``thrust_between_Earth_Moon``: transfer trajectory between the Earth and the Moon that implements a simple thrust guidance scheme.
- ``thrust_satellite_engine``: using a custom class to model the thrust of a satellite.
- ``two_stage_rocket_ascent``: simulation of an ascent trajectory of a two-stage rocket. Implementation of a custom thrust model and hybrid termination condition.

Advanced examples:

- ``reentry_trajectory``: simulation of a reentry flight for the Space Transportation System (STS) and implementation of aerodynamic guidance.
- ``separation_satellites_diff_drag``: shows the effects of differential drag for CubeSats in LEO.
- ``coupled_translational_rotational_dynamics``: using a multi-type propagator to simulate the coupled translational-rotational dynamics of Phobos around Mars.
- ``impact_manifolds_lpo_cr3bp``: setup and propagation of orbits and their invariant manifolds in the circular restricted three body problem (CR3BP) with a polyhedral secondary body.
- ``mga_trajectories``: simulation of Multiple Gravity Assist (MGA) transfer trajectories using high- and low-thrust transfers, as well as deep space maneuvers (DSMs).

### Pygmo

Examples showing how to optimize a problem modelled with Tudatpy via algorithms provided by Pygmo.

- ``himmelblau_optimization``: finds the minimum of an analytical function to show the basic usage of Pygmo
- ``asteroid_orbit_optimization``: simulates the orbit around the Itokawa asteroid and finds the initial state that ensures optimal coverage and close approaches

## Format

The examples are available as both Jupyter Notebooks and raw ``.py`` scripts. The Python scripts are auto-generated from the Jupyter notebooks to ensure consistency.

### Jupyter Notebook

To run these examples, first create the `tudat-space` conda environment to install `tudatpy` and its required dependencies, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

Then, make sure that the `tudat-space` environment is activated:

```bash
conda activate tudat-space
```

Two packages then need to be added to this environment. First, the `notebook` package is needed to run the Jupyter notebooks:

```bash
conda install notebook
```

Then, if you wish to be able to run the `Pygmo` examples, this package also need to be installed:

```bash
conda install pygmo
```

The `tudat-space` environment has to be added to the Jupyter kernel, running the following:

```bash
python -m ipykernel install --user --name=tudat-space
```

Finally, run the following command to start the Jupyter notebooks:

```bash
jupyter notebook
```

### Static code

To run the examples as regular Python files, you can clone this repository, open the examples on your favorite IDE, and install the `tudat-space` conda environment, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

All of the examples, provided as `.py` files, can then be run and edited as you see fit.

Please note that these `.py` files were generated from the Jupyter Notebooks.

### MyBinder

We set up a repository on [MyBinder](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master): this way, you can explore and run the examples online, without having to set up a development environment or installing the tudatpy conda environment. Click on the button below to launch the examples on ``mybinder``:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master)

## Contribute

Contributions to this repository are always welcome.
It is recommended to use the `tudat-examples` conda environment for the development of example applications, as it contains all dependencies for the creation and maintenance of example applications, such as `ipython`, `nbconvert` in addition to `pygmo`.
Simply install the environment using

```bash
conda env create -f environment.yaml
```

and then activate it:

```bash
conda activate tudat-examples
```

The following guidelines should be followed when creating a new example application.

1. Any modification or addition to this set of examples should be made in a personal fork of the current repository. No changes are to be done directly on a local clone of this repo.
2. The example should be written directly on a Jupyter notebook (`.ipynb` file). Then, the following command can be run from the CLI to create a `.py` file with the same code as the notebook file: `jupyter nbconvert --to python mynotebook.ipynb`. Make sure to change `mynotebook` to the name of the notebook file.
3. The markdown blocks are not optimally converted. Thus, once the `.py` file is created as described above, the script `create_scripts.py` is to be executed. This file reformats the markdown blocks in the `.py` files into a more readable look. Sometimes this cleanup is not perfect, so manually check the `.py` file to make sure everything is fine and correct anything that is not.
4. At this point, the example is complete. You are ready to create a pull request from your personal fork to the current repository, and the admins will take it from there.
