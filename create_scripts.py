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
import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

# Other imports
from tqdm import tqdm


class ErrorCatchingArgumentParser(argparse.ArgumentParser):
    """
    Instantiating this class will print the help message and
    exit if an error occurs while parsing the arguments.
    """

    def error(self, message):
        print(f"Error occurred while parsing arguments: {message}\n")
        self.print_help()
        exit(2)


def parse_cli_arguments() -> dict:

    parser = ErrorCatchingArgumentParser(
        description="Create and clean .py files from Jupyter notebooks.",
        exit_on_error=False,
    )

    # either provide a notebook path or use the --all flag
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "notebook_path", nargs="?", help="Path to the notebook to convert to a script"
    )
    group.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Create scripts for all notebooks. Ignores the path argument",
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean the scripts after conversion",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Do not check the scripts for syntax errors",
    )
    parser.add_argument(
        "--no-run", action="store_true", help="Do not run the scripts after conversion"
    )

    args = parser.parse_args()

    return args


# Utilities
def generate_script(notebook, clean_script: bool = True):
    """
    Transform each notebook into a python script using
    the jupyter nbconvert command line utility.
    Use custom clean_py template in conversion if clean_script is True.
    See https://nbconvert.readthedocs.io/en/latest/customizing.html for nbconvert template documentation.
    """

    # convert to python instead of generic script
    # see https://stackoverflow.com/questions/48568388/nbconvert-suddenly-producing-txt-instead-of-py
    command_args = ["jupyter", "nbconvert", notebook, "--to", "python"]

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

    args = parse_cli_arguments()

    if args.all:
        notebooks_to_clean = glob.glob("**/*.ipynb", recursive=True)
    else:
        notebooks_to_clean = [args.notebook_path]

    example_scripts = [
        notebook.replace(".ipynb", ".py") for notebook in notebooks_to_clean
    ]

    """
    Generate Python scripts from Jupyter notebooks
    """
    for notebook in tqdm(notebooks_to_clean):
        generate_script(notebook, (not args.no_clean))

    if not args.no_clean:
        """
        Clean Python scripts
        """
        for example_python_script in example_scripts:
            clean_script(example_python_script)

    if not args.no_check:
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

    if not args.no_run:
        """
        Test python scripts
        """
        for example_python_script in example_scripts:

            print(f"Testing example:               {example_python_script}")

            # Test the example
            subprocess.run(["python", example_python_script])
