Installation
============

From source (recommended)
-------------------------

.. code-block:: bash

   git clone https://github.com/yanxingjianken/pvtend.git
   cd pvtend
   pip install -e ".[dev]"

From PyPI (coming soon)
-----------------------

.. code-block:: bash

   pip install pvtend

With micromamba
--------------

.. code-block:: bash

   micromamba create -f environment.yml
   micromamba activate pvtend_env
   pip install -e ".[dev]"

Dependencies
------------

Core:
  numpy, scipy, xarray, pandas, netCDF4, tqdm

Visualization:
  matplotlib, cartopy

Test:
  pytest

Docs:
  sphinx, sphinx-rtd-theme, nbsphinx
