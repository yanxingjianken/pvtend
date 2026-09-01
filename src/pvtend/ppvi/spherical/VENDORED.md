# Vendored: the global spherical inversion

These modules are a copy, not a fork. They come from

    pv_inversion_spherical, commit 9af88ed
    2026-09-01

**Do not edit them here.** A change made in this copy is invisible to the tests and
the documentation that justify it, and the two copies will disagree without anyone
noticing. Change the source repository, then re-copy the whole directory:

    cp $SRC/src/pvinv_sph/*.py src/pvtend/ppvi/spherical/

and update the commit above. The adapter that connects them to this package is
`ppvi/spherical_engine.py`, which is *not* vendored and is the right place for
anything specific to this pipeline.

The import inside these files is `from .module import ...`, relative, so they work
unchanged as a subpackage.
