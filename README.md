# Tudatpy examples

Welcome to the repository showcasing example applications set up with Tudatpy!

If you want to know more about Tudatpy, please visit the [Tudat website](https://docs.tudat.space/en/latest/).
The website also holds the [examples rendered as notebooks](https://docs.tudat.space/en/latest/_src_getting_started/examples.html).

## Format

The examples are available as both Jupyter Notebooks and raw ``.py`` scripts. The Python scripts are auto-generated from the Jupyter notebooks to ensure consistency, using the ``create_scripts.py`` script in this repo.

### Jupyter Notebook

To run these examples, first create the `tudat-space` conda environment to install `tudatpy` and its required dependencies, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

Then, make sure that the `tudat-space` environment is activated:

```bash
conda activate tudat-space
```

If you wish to be able to run the `Pygmo` examples, this package also need to be installed:

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

### Python Code

To run the examples as regular Python files, you can clone this repository, open the examples on your favorite IDE, and install the `tudat-space` conda environment, as described [here](https://docs.tudat.space/en/latest/_src_getting_started/installation.html).

All of the examples, provided as `.py` files, can then be run and edited as you see fit.

## Content

The examples are organized in different categories.

* **Propagation**: Examples showcasing various aspects of the [state propagation functionality](https://docs.tudat.space/en/latest/_src_user_guide/state_propagation.html) in Tudat, ranging from simple unperturbed orbits, to complex multi-body dynamics, re-entry guidance, etc.
* **Estimation**: Examples showcasing various aspects of the [state estimation functionality](https://docs.tudat.space/en/latest/_src_user_guide/state_estimation.html), from both simulated data and real data, such as astrometric data of asteroids, and radio tracking data of planetary missions.
* **Mission Design**: Examples showcasing the [preliminary mission design functionality](https://docs.tudat.space/en/latest/_src_user_guide/prelim_mission_design.html) in Tudat, which provides (semi-)analytical design of transfer trajectory using both low- and high-thrust
* **Optimization**: Examples showing how to [optimize a problem modelled with Tudatpy](https://docs.tudat.space/en/latest/_src_advanced_topics/optimization_pygmo.html) via algorithms provided by Pygmo.

## Contribute

Contributions to this repository are always welcome.
It is recommended to use the `tudat-examples` conda environment for the development of example applications (created using [this .yaml file](https://github.com/tudat-team/tudatpy-examples/blob/master/environment.yaml)), as it contains all dependencies for the creation and maintenance of example applications, such as `ipython`, `nbconvert` in addition to `pygmo`. However, examples developed using the regular (or develop) conda environment are also most welcome!

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
2. The example should be written directly on a Jupyter notebook (`.ipynb` file).
3. Convert the finished `.ipynb` example to a `.py` file with the `create_scripts.py` CLI utility:
    1. Activate the virtual environment:

        ```bash
        conda activate tudat-examples
        ```

    2. Use the `create_scripts.py` CLI application to convert your notebook:

        ```bash
        python create_scripts.py path/to/your/notebook.ipynb
        ```

        By default, this converts the `.ipynb` notebook to a `.py` file, cleans it, checks for syntax errors and runs it.

    3. Use the `-h` flag to see the available options of the CLI utility. A common set of options is

        ```bash
        python create_scripts.py -a --no-run
        ```

        That converts all `.ipynb` files to `.py` files, cleans and checks them for syntax errors but does not run them.

4. At this point, the example is complete. You are ready to create a pull request from your personal fork to the current repository, and the admins will take it from there.
