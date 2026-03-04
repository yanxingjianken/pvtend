Installation
============

.. image:: https://img.shields.io/pypi/v/pvtend.svg
   :target: https://pypi.org/project/pvtend/
   :alt: PyPI version

From PyPI
---------

The simplest way to install ``pvtend``:

.. code-block:: bash

   pip install pvtend

Or with `uv <https://docs.astral.sh/uv/>`_ (fast, Rust-based installer):

.. code-block:: bash

   uv pip install pvtend

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/yanxingjianken/pvtend.git
   cd pvtend
   pip install -e ".[dev]"

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
