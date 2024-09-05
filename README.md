# Tudatpy examples

Welcome to the repository showcasing example applications set up with Tudatpy!

If you want to know more about Tudatpy, please visit the [Tudat website]([https://tudat-space.readthedocs.io/en/latest/](https://docs.tudat.space/en/latest/)).

## Format

The examples are available as both Jupyter Notebooks and raw ``.py`` scripts. The Python scripts are auto-generated from the Jupyter notebooks to ensure consistency.

### Jupyter Notebook

To run these examples, first create the `tudat-space` conda environment to install `tudatpy` and its required depedencies, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

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

To run the examples as regular Python files, you can clone this repository, open the examples on your favorite IDE, and install the `tudat-space` conda environment, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

All of the examples, provided as `.py` files, can then be run and edited as you see fit.

Please note that these `.py` files were generated from the Jupyter Notebooks.

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

### MyBinder

We set up a repository on [MyBinder](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master): this way, you can explore and run the examples online, without having to set up a development environment or installing the tudatpy conda environment. Click on the button below to 
launch the examples on ``mybinder``:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tudat-team/tudatpy-examples/master)

## Contribute

Contributions to this repository are always welcome. However, there are some guidelines that should be followed when creating a new example application.
Here are some points to be kept in mind.

1. Any modification or addition to this set of examples should be made in a personal fork of the current repository. No changes are to be done directly on a local clone of this repo.
2. The example should be written directly on a Jupyter notebook (.ipynb file). Then, the following command can be run from the CLI to create a .py file with the same code as the notebook file: `jupyter nbconvert --to python mynotebook.ipynb`. Make sure to change `mynotebook` to the name of the notebook file.
3. The markdown blocks are not optimally converted. Thus, once the .py file is created as described above, the script `clean_py_notebooks.py` is to be executed. This file reformats the markdown blocks in the .py files into a more readable look. Sometimes this cleanup is not perfect, so manually check the .py file to make sure everything is fine and correct anything that is not.
4. At this point, the example is complete. You are ready to create a pull request from your personal fork to the current repository, and the admins will take it from there.
