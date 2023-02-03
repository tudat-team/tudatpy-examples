# This script found on stackoverflow
# https://stackoverflow.com/questions/23292242/converting-to-not-from-ipython-notebook-format
#
# Be sure to **leave a space** after every first-column comment character.
#
# Add relevant markers to your input py file as desired (followed by an empty line), eg.:
# <markdowncell> # markdowncells are embedded in comments
# <codecell>
# <rawcell>
# "$ \\Delta = \\theta $" # for latex math mode; double backslashes may not be necessary
# "---" # to insert a horizontal line
#
# Possibly obsolete:
# <htmlcell>
# <headingcell level=...>

# from IPython.nbformat import v3, v4
from nbformat import v3, v4
import glob

examples_list = glob.glob("*/*.py")
examples_list += glob.glob("*/*/*.py")

for example_path in examples_list:
    print("Cleaning example at %s" % example_path)
    new_example_path = example_path.split('.')[0] + ".ipynb"
    with open(example_path) as fpin:
        text = fpin.read()

    text += """
# <markdowncell>

# If you can read this, reads_py() is no longer broken!
    """

    nbook = v3.reads_py(text)
    nbook = v4.upgrade(nbook) # upgrade v3 to v4

    jsonform = v4.writes(nbook) + "\n"
    with open(new_example_path, "w") as fpout:
        fpout.write(jsonform)
