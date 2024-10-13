"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

# PLEASE NOTE:
# This script is NOT a tudatpy example.
# It is a script to clean the .py files that are generated from Jupyter notebooks, using the following option: File > Download as > Python (.py)
# Running it will automatically edit all the .py example files (please check the changes made before pushing them to the repository).

import glob
# Standard library imports
import re
import subprocess

# Other imports
from tqdm import tqdm


# Utilities
def request_confirmation(message):
    message = f"{message} [y/N]"
    width = max(60, len(message) + 20)
    return (
        input(f'{"="*width}\n{message:^{width}}\n{"="*width}\n').strip().lower() == "y"
    )


def generate_script(notebook):
    """
    Transform each notebook into a python script using
    the jupyter nbconvert command line utility
    """
    subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook])


if __name__ == "__main__":

    """
    Use the find command line utility to find the paths of all
    notebooks in this repository and store them in a list
    """
    example_notebooks = glob.glob("**/*.ipynb", recursive=True)
    example_scripts = [
        notebook.replace(".ipynb", ".py") for notebook in example_notebooks
    ]
    all_python_files = glob.glob("**/*.py", recursive=True)

    if request_confirmation("Regenerate Python scripts from Jupyter notebooks?"):
        # Generate the python scripts
        for notebook in tqdm(example_notebooks):
            generate_script(notebook)
        # Assert that all the notebooks were converted to python scripts
        assert all([script in all_python_files for script in example_scripts]), (
            f"Unsuccessful: not all notebooks were converted to python scripts. Failed conversions:\n"
            + "\n".join(
                [script for script in example_scripts if script not in all_python_files]
            )
        )
    else:
        # If there are missing scripts
        if not all([script in all_python_files for script in example_scripts]):
            # Generate the missing python scripts
            for script in [
                script for script in example_scripts if script not in all_python_files
            ]:
                generate_script(script)

    """
    Clean up the python scripts
    """

    for example_python_script in example_scripts:

        print(f"Cleaning example:              {example_python_script}")

        with open(example_python_script, "r+") as file:

            # Read example
            example_content = file.readlines()

            # Remove file type and encoding
            if "!/usr/bin/env python" in example_content[0]:
                example_content = example_content[3:]

            # State
            checking_comment_end = False
            skip_next = False

            # Indentation
            indentation = ""

            # Go trough each line in the example
            for i, line in enumerate(example_content):

                if skip_next:
                    skip_next = False
                    continue

                # --> Remove the "In[x]" notebook inputs
                if "In[" in line:
                    # Also remove the two lines after
                    [example_content.pop(i) for _ in range(3)]

                # --> End of MD cell
                elif checking_comment_end:
                    # --> End of cell: if the line is empty, the markdown cell is finished
                    if line == "\n":
                        # Add """ to close the string comment, then an empty line
                        example_content[i] = '"""\n'
                        example_content.insert(i + 1, "\n")
                        checking_comment_end = False
                    # --> Second title: detect if we have a second title in the same markdown cell, and mark the separation
                    elif "##" in line:
                        example_content[i] = line.replace("# ", "", 1)
                        example_content.insert(i, '"""\n\n')
                        example_content.insert(i + 2, '"""\n')
                        skip_next = True
                    # If we are still in the markdown cell, remove the simple # that indicates a comment line
                    else:
                        example_content[i] = line.replace("# ", "", 1)

                # --> Start of MD cell: if the line starts with # #, we are in a markdown cell that starts with a title
                elif "# #" in line:
                    # Replace the first line to keep the title with a comment #
                    example_content[i] = line.replace("# ", "", 1)
                    example_content.insert(i + 1, '"""\n')
                    if example_content[i + 2] == "\n":
                        example_content.pop(i + 2)
                    checking_comment_end = True  # Start looking for the end of the cell

                # --> Remove the lines that made the plots interactive
                elif "# make plots interactive" in line:
                    [example_content.pop(i) for _ in range(2)]

                # We're in a code cell, so we record the indentation level
                else:

                    # Retrieve the last non-empty line
                    last_nonempty_line = next(
                        line
                        for line in example_content[i - 1 :: -1]
                        if re.match(r"^( {4}){0,2}\S", line)
                    )

                    # Keep track of current indentation
                    indentation = " " * (
                        len(last_nonempty_line) - len(last_nonempty_line.lstrip())
                    )

            file.seek(0)
            file.writelines(example_content)
            file.truncate()

    """
    Check Python scripts for syntax errors
    """
    print("\nChecking Python scripts for syntax errors\n")
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
    print("")

    """
    Test python scripts
    """
    if request_confirmation("Test generated Python scripts?"):
        for example_python_script in example_scripts:

            print(f"Testing example:               {example_python_script}")

            # Test the example
            subprocess.run(["python", example_python_script])
