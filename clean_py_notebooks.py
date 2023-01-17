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

examples_list = glob.glob("*/*.py")
examples_list += glob.glob("*/*/*.py")

for example_path in examples_list:
    print("Cleaning example at %s" % example_path)

    with open(example_path, "r+") as file:
        example_content = file.readlines()

        # Remove file type and encoding
        if "!/usr/bin/env python" in example_content[0]:
            example_content = example_content[3:]

        checking_comment_end = False
        skip_next = False
        # Go trough each line in the example
        for i, line in enumerate(example_content):
            if skip_next:
                skip_next = False
                continue
            # Remove the "In[x]" notebook inputs    
            if "In[" in line:
                # Also remove the two lines after
                [example_content.pop(i) for _ in range(3)]

            # Check for the end of a markdown cell
            elif checking_comment_end:
                # If the line is empty, the markdown cell is finished
                if line == "\n":
                    # Add """ to close the string comment, then an empty line
                    example_content[i] = "\"\"\"\n"
                    example_content.insert(i+1, "\n")
                    checking_comment_end = False
                # Detect if we have a second title in the same markdown cell, and mark the separation
                elif "##" in line:
                    example_content[i] = line.replace("# ", "", 1)
                    example_content.insert(i-1, "\"\"\"\n\n")
                    example_content.insert(i+2, "\"\"\"\n")
                    skip_next = True
                # If we are still in the markdown cell, remove the simple # that indicates a comment line
                else:
                    example_content[i] = line.replace("# ", "", 1)

            # If the line starts with # #, we are in a markdown cell that starts with a title
            elif "# #" in line:
                # Replace the first line to keep the title with a comment #
                example_content[i] = line.replace("# ", "", 1)
                example_content.insert(i+1, "\"\"\"\n")
                checking_comment_end = True # Start looking for the end of the cell

            # Remove the lines that made the plots interactive
            elif "# make plots interactive" in line:
                [example_content.pop(i) for _ in range(2)]
                

        file.seek(0)
        file.writelines(example_content)
        file.truncate()
