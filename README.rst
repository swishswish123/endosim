endoSim
===============================

.. image:: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim
   :alt: Logo

.. image:: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/badges/master/build.svg
   :target: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/pipelines
   :alt: GitLab-CI test status

.. image:: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/badges/master/coverage.svg
    :target: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/endoSim/badge/?version=latest
    :target: http://endoSim.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Aure Enkaoua

endoSim is part of the `scikit-surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

endoSim supports Python 2.7 and Python 3.6.

endoSim is currently a demo project, which will add/multiply two numbers. Example usage:

::

    python endosim.py 5 8
    python endosim.py 3 6 --multiply

Please explore the project structure, and implement your own functionality.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc endosim


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2022 University College London.
endoSim is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim
.. _`Documentation`: https://endoSim.readthedocs.io
.. _`scikit-surgery`: https://github.com/SciKit-Surgery/scikit-surgery
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/blob/master/CONTRIBUTING.rst
.. _`license file`: https://weisslab.cs.ucl.ac.uk/COMPASS/software/endoSim/blob/master/LICENSE

