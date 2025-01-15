.. mhn_doc documentation master file, created by
   sphinx-quickstart on Fri Dec  2 20:46:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################################################################
*mhn*: A Python Package for Analyzing Cancer Progression with Mutual Hazard Networks
####################################################################################

Mutual Hazard Networks (MHN) were first introduced by `Schill et al. (2019) <https://academic.oup.com/bioinformatics/article/36/1/241/5524604>`_
and are used to model cancer progression.
This Python package provides tools to work with MHNs. It includes
:ref:`optimizer classes<Available Optimizers in the *optimizers* module>`
that enable training an MHN with just a few lines of code. Additionally, it
offers utility functions such as plotting MHNs and generating artificial tumor
histories for a given MHN. The package also incorporates state space restriction,
allowing the training of MHNs with well over 100 events, provided that individual
samples contain no more than about 25 active events.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   mhn


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
