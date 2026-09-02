"""Tests for the planetary/eddy scale split of a PV anomaly.

The split has to be exactly additive and needs no tuning; the object it draws
has to be the tracked event and not whatever is deepest in the box; and the
wrap seam, where a pass over the whole circle leaves its two flagged columns
adjacent, must not be read as planetary amplitude.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestScaleSplit:
    """The split must be exactly additive, and must not need tuning."""

    @staticmethod
    def _field(seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        NL, NY, NX = 9, 60, 288
        q = rng.standard_normal((NL, NY, NX)) * 0.4
        jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
        blob = -3.0 * np.exp(-(((jj - 25) / 6.) ** 2
                               + (((ii - 100 + 144) % 288 - 144) / 10.) ** 2))
        q[4:9] += blob
        th = rng.standard_normal((NY, NX)) * 2.0
        return q, th

    def test_pieces_sum_back_exactly(self):
        """Pass D is linear in its source, so the split is only legitimate if
        the pieces sum to the unsplit anomaly to machine precision."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field()
        upper, top = [4, 5, 6, 7], 8
        out = ss.split_at_box_minimum(q, th, upper, top,
                                      np.arange(5, 55), np.arange(60, 150))
        np.testing.assert_allclose((out["q_p"] + out["q_e"])[upper], q[upper],
                                   atol=1e-12)
        np.testing.assert_allclose(out["th_p"] + out["th_e"], th, atol=1e-12)

    def test_box_minimum_seed_never_needs_a_threshold(self):
        """The box holds a tracked anticyclone, so its minimum is negative and
        the flood fill is unconditional — there is no threshold to tune."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=3)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)
        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        assert out["seed_source"] == "box_min"
        lev, j, i = out["seed"]
        assert out["mask"][lev, j, i], "the seed must lie inside its own object"

    def test_object_cannot_leave_the_box(self):
        """Cropping before the flood fill is what stops the k<=4 negative side
        linking through the subtropics into a hemispheric blob."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=5)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)
        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        outside = np.ones(q.shape[-1], bool)
        outside[bo] = False
        assert not out["mask"][..., outside].any()

    def test_planetary_part_is_wavenumber_limited(self):
        """The re-filter after masking is the point: q'_k4 * M is not itself
        k<=4, because a sharp mask edge puts power back into every wavenumber."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=7)
        upper, top = [4, 5, 6, 7], 8
        out = ss.split_at_box_minimum(q, th, upper, top,
                                      np.arange(5, 55), np.arange(60, 150))
        F = np.abs(np.fft.rfft(out["q_p"][upper], axis=-1)) ** 2
        keep = F[..., ss.KMIN:ss.KMAX + 1].sum()
        assert keep / F.sum() > 0.999

    def test_contour_is_a_fraction_of_the_events_own_minimum(self):
        """The contour scales with the event, and the seed stays inside it.

        `frac * q_min` is negative and `q_min < frac * q_min` for any frac < 1,
        so the box minimum is inside its own contour by construction -- which is
        what keeps the flood fill unconditional after adding a threshold back.
        """
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=11)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)

        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        assert out["thresh_used"] == pytest.approx(ss.OBJ_FRAC * out["q_min"])
        assert out["mask"][out["seed"]]

        # a tighter contour must give a strictly smaller object
        tight = ss.split_at_box_minimum(q, th, upper, top, bl, bo, frac=0.7)
        loose = ss.split_at_box_minimum(q, th, upper, top, bl, bo, frac=0.0)
        assert tight["mask"].sum() < out["mask"].sum() < loose["mask"].sum()


# ── the wrap seam ────────────────────────────────────────────────────
class TestDatelineBoxIsContiguous:
    """`split_at_box_minimum` floods with ndimage.label, which is not periodic.
    A box spanning the dateline must therefore arrive ordered around the circle,
    not sorted by index, or the object is cut in half and a false adjacency is
    created between longitudes 240 deg apart.
    """

    NLON, DLON, LON_HALF = 288, 1.25, 60.0

    def _box(self, clon):
        from pvtend.tendency import _circ_nearest_lon, _wrapped_lon_index
        lon = np.arange(-180.0, 180.0, self.DLON)
        return lon, _wrapped_lon_index(
            _circ_nearest_lon(lon, clon),
            LON_PAD=int(round(self.LON_HALF / self.DLON)), nlon=self.NLON)

    def test_box_is_circularly_contiguous_at_the_dateline(self):
        _, box = self._box(175.0)
        step = np.diff(box) % self.NLON
        assert set(np.unique(step)) == {1}, "columns must be adjacent on the circle"

    def test_box_holds_the_right_longitudes(self):
        lon, box = self._box(175.0)
        d = np.abs((lon[box] - 175.0 + 180.0) % 360.0 - 180.0)
        assert d.max() <= self.LON_HALF + 0.5 * self.DLON

    def test_a_seam_crossing_object_stays_one_component(self):
        from pvtend.ppvi.scale_split import component_containing
        lon, box = self._box(178.0)
        q = np.zeros((1, 3, self.NLON))
        blob = np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= 10.0
        q[:, :, blob] = -1.0
        sub = q[:, :, box] < -0.5
        seed = int(np.nonzero(box == int(np.nonzero(blob)[0][len(np.nonzero(blob)[0]) // 2]))[0][0])
        comp = component_containing(sub, 0, 1, seed)
        assert comp.sum() == sub.sum(), "the blob must not be split by the seam"

    def test_sorted_box_would_have_split_it(self):
        """Guard the guard: the old boolean-mask ordering really does fail."""
        from pvtend.ppvi.scale_split import component_containing
        lon = np.arange(-180.0, 180.0, self.DLON)
        old_box = np.nonzero(
            np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= self.LON_HALF)[0]
        q = np.zeros((1, 3, self.NLON))
        blob = np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= 10.0
        q[:, :, blob] = -1.0
        sub = q[:, :, old_box] < -0.5
        seed = int(np.nonzero(old_box == int(np.nonzero(blob)[0][0]))[0][0])
        comp = component_containing(sub, 0, 1, seed)
        assert comp.sum() < sub.sum(), "expected the old ordering to split the object"


class TestSeedNearCentre:
    """The object must be the TRACKED feature, not whatever is deepest in the
    +/-60 x +/-30 box. Measured on 30+30 m091 events: a blocking high IS its box
    minimum, a propagating high is not, and the old policy then inverted an
    unrelated system a median 4193 km away while every sum still balanced.
    """

    NL, NY, NX = 9, 40, 288
    UPPER, TOP = [4, 5, 6, 7], 8

    def _field(self):
        """A shallow anomaly at the tracked centre, a DEEPER one far away."""
        q = np.zeros((self.NL, self.NY, self.NX))
        lon = np.arange(self.NX) * (360.0 / self.NX) - 180.0
        for k in self.UPPER:
            for j in range(self.NY):
                # the event: modest, centred at lon 0, row 20
                q[k] += -60.0 * np.exp(-(((lon - 0.0) / 18.0) ** 2))[None, :] \
                    * np.exp(-(((np.arange(self.NY) - 20) / 5.0) ** 2))[:, None]
                # a deeper distractor 100 deg away, inside the same box
                q[k] += -200.0 * np.exp(-(((lon + 100.0) / 18.0) ** 2))[None, :] \
                    * np.exp(-(((np.arange(self.NY) - 22) / 5.0) ** 2))[:, None]
                break
        for k in self.UPPER[1:]:
            q[k] = q[self.UPPER[0]]
        return q

    @property
    def _box(self):
        return np.arange(5, 35), np.arange(self.NX)   # box holds both features

    def test_box_min_seed_lands_on_the_distractor(self):
        """Guard the guard: the old policy really does pick the wrong feature."""
        from pvtend.ppvi.scale_split import seed_from_box_min, zonal_filter
        q = self._field(); bl, bo = self._box
        _, _, i = seed_from_box_min(zonal_filter(q, 1, 4), self.UPPER, bl, bo)
        lon_i = i * (360.0 / self.NX) - 180.0
        assert abs(lon_i + 100.0) < 25.0, f"expected the distractor, got lon {lon_i}"

    def test_near_centre_seed_lands_on_the_event(self):
        from pvtend.ppvi.scale_split import seed_near_centre, zonal_filter
        q = self._field(); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        _, _, i = seed_near_centre(zonal_filter(q, 1, 4), self.UPPER, bl, bo,
                                   centre_lat=20, centre_lon=ic,
                                   halo_lat=10, halo_lon=16)
        lon_i = i * (360.0 / self.NX) - 180.0
        assert abs(lon_i) < 25.0, f"expected the tracked event, got lon {lon_i}"

    def test_contour_scales_to_the_LOCAL_minimum(self):
        """0.35 x box min would be deeper than the event and reject it."""
        from pvtend.ppvi.scale_split import split_near_centre, split_at_box_minimum
        q = self._field(); bl, bo = self._box
        th = np.zeros((self.NY, self.NX))
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        near = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                 centre_lat=20, centre_lon=ic,
                                 halo_lat=10, halo_lon=16)
        boxm = split_at_box_minimum(q, th, self.UPPER, self.TOP, bl, bo)
        assert abs(near["q_min"]) < abs(boxm["q_min"]), "local min must be shallower"
        assert near["seed_source"] == "near_centre"

    def test_mask_covers_the_tracked_centre(self):
        from pvtend.ppvi.scale_split import split_near_centre
        q = self._field(); th = np.zeros((self.NY, self.NX)); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        out = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                centre_lat=20, centre_lon=ic,
                                halo_lat=10, halo_lon=16)
        assert out["mask"][self.UPPER][:, 20, ic].any()

    def test_pieces_still_sum_to_the_total(self):
        from pvtend.ppvi.scale_split import split_near_centre
        q = self._field(); th = np.zeros((self.NY, self.NX)); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        out = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                centre_lat=20, centre_lon=ic,
                                halo_lat=10, halo_lon=16)
        np.testing.assert_allclose(out["q_p"] + out["q_e"], q, atol=1e-9)

    def test_centre_outside_the_box_is_refused(self):
        from pvtend.ppvi.scale_split import seed_near_centre, zonal_filter
        q = self._field(); bl, bo = self._box
        with pytest.raises(ValueError, match="not inside the event box"):
            seed_near_centre(zonal_filter(q, 1, 4), self.UPPER, bl, bo,
                             centre_lat=999, centre_lon=0,
                             halo_lat=10, halo_lon=16)

    def test_no_negative_anomaly_near_the_centre_is_refused(self):
        from pvtend.ppvi.scale_split import seed_near_centre
        q = np.ones((self.NL, self.NY, self.NX)) * 5.0
        bl, bo = self._box
        with pytest.raises(ValueError, match="no negative"):
            seed_near_centre(q, self.UPPER, bl, bo, centre_lat=20, centre_lon=10,
                             halo_lat=4, halo_lon=4)

