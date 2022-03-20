# Tudatpy examples

Welcome to the repository showcasing example applications set up with Tudatpy!

If you want to know more about Tudatpy, please visit the [Tudat Space](https://tudat-space.readthedocs.io/en/latest/).

## Format

The examples are available as both Jupyter Notebooks and raw ``.py`` scripts.

### MyBinder

We set up a repository on [MyBinder](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master): this way, you can explore and run the examples online, without having to set up a development environment or installing the tudatpy conda environment. Click on the button below to 
launch the examples on ``mybinder``:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master)

### Jupyter Notebook
You can alternatively run the Jupyter notebooks directly on your computer.
To do so, first create the `tudat-space` conda environment to install `tudatpy` and its required depedencies.

A detailed guide on how to do this can be found in the [tudatpy user guide](https://tudat-space.readthedocs.io/en/latest/_src_getting_started/installation.html).

Then, make sure that the `tudat-space` environment is activated:
````
conda activate tudat-space
````

Two packages then need to be added to this environment. First, the `notebook` package is needed to run the Jupyter notebooks:
````
conda install notebook
````

Then, if you wish to be able to run the `Pygmo` examples, this package also need to be installed:
````
conda install pygmo
````

The `tudat-space` environment has to be added to the Jupyter kernel, running the following:

````
python -m ipykernel install --user --name=tudat-space
````

Finally, run the following command to start the Jupyter notebooks:
````
jupyter notebook
````

### Static code

Otherwise, you can clone this repository, open the examples on your favorite IDE, and install the `tudat-space` conda environment.

More instructions are reported in the guides hosted on [Tudat Space](https://tudat-space.readthedocs.io/en/latest/).

All of the examples, provided as `.py` files, can then be run and edited as you see fit.

Please note that these `.py` files were generated from the Jupyter Notebooks.
The clarity and format of the code may suffer from this, and we advise to run the notebooks for a simpler user experience.

## Content

The examples are organized in different categories.

### Estimation

Examples related to state estimation.

- ``basic_orbit_estimation``: orbit estimation of a satellite in MEO

### Propagation

Examples related to state propagation.

- ``keplerian_satellite_orbit``: simulation of a Keplerian orbit around Earth (two-body problem)
- ``perturbed_satellite_orbit``: simulation of a perturbed orbit around Earth
- ``linear_sensitivity_analysis``: extension of the ``perturbed_satellite_orbit`` example to propagate variational 
  equations to perform a sensitivity analysis
- ``reentry_trajectory``: simulation of a reentry flight for the Space Transportation System (STS) and 
  implementation of aerodynamic guidance
- ``solar_system_propagation``: numerical propagation of solar-system bodies, showing how a hierarchical, multi-body 
  simulation  can be set up
- ``thrust_between_Earth_Moon``: transfer trajectory between the Earth and the Moon that implements a simple 
  thrust guidance scheme
- ``two_satellite_phasing``: shows the effects of differential drag for CubeSats in LEO

### Pygmo

Examples showing how to optimize a problem modelled with Tudatpy via algorithms provided by Pygmo.

- ``himmelblau_optimization``: finds the minimum of an analytical function to show the basic usage of Pygmo
- ``asteroid_orbit_optimization``: simulates the orbit around the Itokawa asteroid and finds the initial state that 
  ensures optimal coverage and close approaches
