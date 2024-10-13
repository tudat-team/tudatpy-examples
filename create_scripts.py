"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

# PLEASE NOTE:
# This script is NOT a tudatpy example.
# It is a script to create and clean .py files from Jupyter notebooks.
# See python create_scripts.py -h for more information.
# Running it will automatically edit all the .py example files (please check the changes made before pushing them to the repository).

# Standard library imports
import glob
import re
import subprocess
import sys
from pathlib import Path

# Other imports
from tqdm import tqdm


def usage() -> None:
    """Print usage information for the script."""

    print(
        "Usage: python create_scripts.py path/to/notebook.ipynb [OPTIONS]", end="\n\n"
    )
    print("Arguments:")
    print("  path/to/notebook.ipynb      Path to the notebook to convert to a script")
    print("Options:")
    print("  -h | --help                 Show this message")
    print(
        "  -a | --all                  Create scripts for all notebooks. Ignores the path argument"
    )
    print("  --no-clean                  Do not clean the scripts after conversion")
    print("  --no-check                  Do not check the scripts for syntax errors")
    print("  --no-run                    Do not run the scripts after conversion")


def parse_cli_arguments(cli_args: list[str]) -> dict:

    # default arguments
    ARGUMENTS = {
        "NOTEBOOKS": None,
        "CLEAN_SCRIPTS": True,
        "CHECK_SCRIPTS": True,
        "RUN_SCRIPTS": True,
    }

    if len(cli_args) < 2:
        print("Please provide a notebook file to convert.")
        usage()
        exit(1)

    # Parse input
    args = iter(cli_args[1:])
    for ii, arg in enumerate(args):
        if ii == 0:
            if Path(arg).resolve().exists():
                ARGUMENTS["NOTEBOOKS"] = [arg]
                continue
        if arg in ("-h", "--help"):
            usage()
            exit(0)
        elif arg in ("-a", "--all"):
            ARGUMENTS["NOTEBOOKS"] = glob.glob("**/*.ipynb", recursive=True)
        elif arg == "--no-clean":
            ARGUMENTS["CLEAN_SCRIPTS"] = False
        elif arg == "--no-check":
            ARGUMENTS["CHECK_SCRIPTS"] = False
        elif arg == "--no-run":
            ARGUMENTS["RUN_SCRIPTS"] = False
        else:
            print("Invalid command")
            usage()
            exit(1)

    return ARGUMENTS


# Utilities
def generate_script(notebook, clean_script: bool = True):
    """
    Transform each notebook into a python script using
    the jupyter nbconvert command line utility.
    Use custom clean_py template in conversion if clean_script is True.
    See https://nbconvert.readthedocs.io/en/latest/customizing.html for nbconvert template documentation.
    """
    command_args = ["jupyter", "nbconvert", notebook, "--to", "script"]

    if clean_script:
        command_args += ["--template", "templates/clean_py"]

    subprocess.run(command_args)


def clean_script(script):
    print(f"Cleaning example:              {script}")

    with open(script, "r+") as file:

        # Read example
        example_content = file.readlines()

        if "plt.show()" not in example_content:
            example_content.append("\nplt.show()")

        file.seek(0)
        file.writelines(example_content)
        file.truncate()


if __name__ == "__main__":

    cli_args = sys.argv

    ARGUMENTS = parse_cli_arguments(cli_args)

    example_notebooks = ARGUMENTS["NOTEBOOKS"]
    example_scripts = [
        notebook.replace(".ipynb", ".py") for notebook in example_notebooks
    ]

    """
    Generate Python scripts from Jupyter notebooks
    """
    for notebook in tqdm(example_notebooks):
        generate_script(notebook, ARGUMENTS["CLEAN_SCRIPTS"])

    if ARGUMENTS["CLEAN_SCRIPTS"]:
        """
        Clean Python scripts
        """
        for example_python_script in example_scripts:
            clean_script(example_python_script)

    if ARGUMENTS["CHECK_SCRIPTS"]:
        """
        Check Python scripts for syntax errors
        """
        NO_SYNTAX_ERRORS = True

        print("\nChecking Python scripts for syntax errors...\n")

        for example_python_script in example_scripts:

            # Test the example
            result = subprocess.run(
                ["python", "-m", "py_compile", example_python_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            if result.returncode != 0:
                print(f"Unsuccessful: syntax error in example: {example_python_script}")
                example_scripts.remove(example_python_script)
                NO_SYNTAX_ERRORS = False

        if NO_SYNTAX_ERRORS:
            print("All examples are free of syntax errors.")

        print("")

    if ARGUMENTS["RUN_SCRIPTS"]:
        """
        Test python scripts
        """
        for example_python_script in example_scripts:

            print(f"Testing example:               {example_python_script}")

            # Test the example
            subprocess.run(["python", example_python_script])
