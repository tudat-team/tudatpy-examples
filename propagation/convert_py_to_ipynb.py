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

with open("inputfile.py") as fpin:
    text = fpin.read()

text += """
# <markdowncell>

# If you can read this, reads_py() is no longer broken!
"""

nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook) # upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"
with open("outputfile.ipynb", "w") as fpout:
    fpout.write(jsonform)
