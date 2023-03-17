.. mhn_doc documentation master file, created by
   sphinx-quickstart on Fri Dec  2 20:46:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###################################################################
mhn: A Python Package to efficiently compute Mutual Hazard Networks
###################################################################

Mutual Hazard Networks (MHN) were first introduced by `Schill et al. (2019) <https://academic.oup.com/bioinformatics/article/36/1/241/5524604>`_
and are used to model cancer progression.
This Python package can be used to work with MHNs. It includes functions that were part of the
original R implementation as well as functions that make use of state-space restriction
to make learning a new MHN from cancer data faster and more efficient. Furthermore, it
also contains functions to work with data for which the samples' ages are known and can
therefore be considered while learning an MHN (see `Rupp et al. (2021) <https://arxiv.org/abs/2112.10971>`_).
There are :ref:`optimizer classes<Available Optimizers in the *optimizers* module>` for data with known sample ages as well as for data without, which make learning a new MHN possible with
only a few lines of code.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   mhn
   original:   functions working on a full state-space <original/mhn.original>
   ssr:        functions making use of state-space restriction <ssr/mhn.ssr>
   optimizers: the Goto-module to learn new MHNs from data <optimizers>
   model:      a module that contains the MHN class <model>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

: