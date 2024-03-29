from operator import itemgetter
from collections import namedtuple, deque

import numpy as np

from utils.math_utils import Vector2D, Vector3D, EPSILON, pi
from utils.utils import documentation_inheritance


def angle_normalize(angle):
    """
    Normalize angle to [0, 2*pi).

    Parameters
    ----------
    angle : float

    Returns
    -------
    float
    """
    return angle % (2.0 * pi)


def qe_solve(a, k, c):
    """
    Solves quadratic equation 'ax^2 + 2kx + c = 0'.

    Parameters
    ----------
    a : float
    k : float
    c : float

    Returns
    -------
    Generator[float]
        Generator of solutions (if there are no solutions generator will be empty).
    """
    d = k ** 2 - a * c
    if d < 0:
        return
    if d < EPSILON:
        yield -k / a
    else:
        sqrt_d = np.sqrt(d)
        yield (-k + sqrt_d) / a
        yield (-k - sqrt_d) / a


class FullCircleSegment:
    """ A segment of plasma given by 'r_min <= r <= r_max' and 'z_min <= z <= z_max'. """

    def __init__(self, r_min, r_max, z_min, z_max, lum=0.0):
        """
        Parameters
        ----------
        r_min : float
        r_max : float
        z_min : float
        z_max : float
        lum : float, optional
            Luminosity of the segment.
        """
        if r_min > r_max or z_min > z_max:
            raise ValueError("Arguments must satisfy: 'r_min <= r_max' and 'z_min <= z_max'")
        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max
        self.lum = lum

    def __repr__(self):
        state = 'r_min={}, r_max={}, z_min={}, z_max={}'
        return state.format(self.r_min, self.r_max, self.z_min, self.z_max)

    def intersection_points(self, p, e):
        """
        Finds intersections of segment with line 'q = p + te'.

        Parameters
        ----------
        p : Vector3D
        e : Vector3D

        Returns
        -------
        Generator[(float, Vector3D)]
            Generator of pairs (t, q) - intersection points sorted ascending by 't'.
        """
        points = []
        for z in (self.z_min, self.z_max):
            points.extend(self._z_surface_intersection(z, p, e))
        points.extend(self._r_surface_intersection(self.r_max, p, e))
        if self.r_min > EPSILON:
            points.extend(self._r_surface_intersection(self.r_min, p, e))
        return self._points_filter(points)

    def intersection_length(self, p, e):
        """
        Finds the summary length of line 'q = p + te' sections which are inside the segment.

        Parameters
        ----------
        p : Vector3D
        e : Vector3D

        Returns
        -------
        float
        """
        points = list(self.intersection_points(p, e))
        n = len(points)
        if n == 0 or n == 1:
            return 0.0
        elif n == 2:
            return points[1][0] - points[0][0]
        elif n == 4:
            return points[3][0] - points[2][0] + points[1][0] - points[0][0]
        else:
            raise NotImplementedError('{} intersection points found: {}'.format(n, points))

    def _r_surface_intersection(self, r, p, e):
        a = e.x ** 2 + e.y ** 2
        k = p.x * e.x + p.y * e.y
        c = p.x ** 2 + p.y ** 2 - r ** 2
        for t in qe_solve(a, k, c):
            q = p + t * e
            if self.z_min <= q.z <= self.z_max:
                yield t, q

    def _z_surface_intersection(self, z, p, e):
        if abs(e.z) < EPSILON:
            return
        t = (z - p.z) / e.z
        q = p + t * e
        if self.r_min <= q.r <= self.r_max:
            yield t, q

    @staticmethod
    def _points_filter(points):
        points.sort(key=itemgetter(0))
        if len(points) > 0:
            yield points[0]
        for i in range(1, len(points)):
            if abs(points[i][0] - points[i - 1][0]) > EPSILON:
                yield points[i]


class PhiBoundedSegment(FullCircleSegment):
    """
    A segment of plasma given by 'r_min <= r <= r_max',
    'z_min <= z <= z_max' and 'phi_min <= phi <= phi_max'.
    """

    def __init__(self, r_min, r_max, phi_min, phi_max, z_min, z_max, lum=0.0):
        """
        Parameters
        ----------
        r_min : float
        r_max : float
        phi_min : float
        phi_max : float
        z_min : float
        z_max : float
        lum : float, optional
            Luminosity of the segment.
        """
        super().__init__(r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max, lum=lum)
        if phi_min > phi_max:
            raise ValueError("Arguments must satisfy: 'phi_min <= phi_max'")
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.rotate(0.0)

    def __repr__(self):
        state = 'r_min={}, r_max={}, phi_min={}, phi_max={}, z_min={}, z_max={}'
        return state.format(self.r_min, self.r_max, self.phi_min, self.phi_max, self.z_min, self.z_max)

    def rotate(self, phi):
        """
        Rotates the segment (anticlockwise looking from the top).

        Parameters
        ----------
        phi : float
            Angle of rotation.
        """
        phi = angle_normalize(phi)
        self.phi_min += phi
        self.phi_max += phi
        if self.phi_min >= 2.0 * pi:
            self.phi_min = angle_normalize(self.phi_min)
            self.phi_max = angle_normalize(self.phi_max)
        assert self.phi_min <= self.phi_max

    @documentation_inheritance(FullCircleSegment.intersection_points)
    def intersection_points(self, p, e):
        points = []
        for z in (self.z_min, self.z_max):
            points.extend(self._z_surface_intersection(z, p, e))
        for phi in (self.phi_min, self.phi_max):
            points.extend(self._phi_surface_intersection(phi, p, e))
        for r in (self.r_min, self.r_max):
            points.extend(self._r_surface_intersection(r, p, e))
        return self._points_filter(points)

    def _r_surface_intersection(self, r, p, e):
        for t, q in super()._r_surface_intersection(r, p, e):
            if self._phi_bounds_check(q.phi):
                yield t, q

    def _z_surface_intersection(self, z, p, e):
        for t, q in super()._z_surface_intersection(z, p, e):
            if self._phi_bounds_check(q.phi):
                yield t, q

    def _phi_surface_intersection(self, phi, p, e):
        phi = angle_normalize(phi)
        n = Vector2D(np.sin(phi), -np.cos(phi))
        e_dot_n = e.xy.dot(n)
        if abs(e_dot_n) < EPSILON:
            # e is parallel to plane
            return
        t = - p.xy.dot(n) / e_dot_n
        q = p + t * e
        if EPSILON < abs(phi - q.phi) < 2.0 * pi - EPSILON:
            # opposite half-plane
            return
        if self.z_min <= q.z <= self.z_max and self.r_min <= q.r <= self.r_max:
            yield t, q

    def _phi_bounds_check(self, phi):
        phi = angle_normalize(phi)
        if self.phi_max <= 2.0 * pi:
            return self.phi_min <= phi <= self.phi_max
        return self.phi_min <= phi <= 2.0 * pi or 0.0 <= phi <= angle_normalize(self.phi_max)


class CylindricalPlasma:
    """
    Plasma is a cylinder ring given by 'r_min <= r <= r_max' and 'z_min <= z <= z_max'.
    The plasma is always placed in the origin of coordinates, the cylinder axis is aligned with OZ.

    Attributes:
    -----------
        - segments - list of segments
        - phi_0
        - plasma_metrics
        - grid_metrics
        - n_segments
        - n_pol_rings
        - lum
    """
    PlasmaMetrics = namedtuple('PlasmaMetrics', ['r_min', 'r_max', 'z_min', 'z_max'])
    GridMetrics = namedtuple('GridMetrics', ['n_r', 'n_phi', 'n_z', 'd_r', 'd_phi', 'd_z'])

    def __init__(self, r_min=0.0, r_max=1.0, z_min=-1.0, z_max=1.0):
        """
        Parameters
        ----------
        r_min : float, optional
        r_max : float, optional
        z_min : float, optional
        z_max : float, optional
        """
        if r_min > r_max or z_min > z_max:
            raise ValueError("Arguments must satisfy: 'r_min <= r_max' and 'z_min <= z_max'")
        self._plasma_metrics = self.PlasmaMetrics(r_min, r_max, z_min, z_max)
        self._grid_metrics = None
        self.segments = []
        self._phi_0 = 0
        self._rotation_status = None
        self.build_segmentation(n_r=1, n_phi=1, n_z=1)

    def __repr__(self):
        state = 'n_segments={}, phi0={}\n'.format(self.n_segments, self.phi_0)
        state += self.plasma_metrics.__repr__() + '\n'
        state += self.grid_metrics.__repr__() + '\n'
        state += 'pol_rotation_state={}'.format(self._rotation_status)
        return state

    @property
    def phi_0(self):
        """
        The angle through which the plasma is rotated in comparison with the initial state
        (phi_0 = 0 by default, if no 'rotate()' was called).

        Returns
        -------
        float

        See also
        --------
        CylindricalPlasma.rotate
        """
        return self._phi_0

    @property
    def plasma_metrics(self):
        """
        Returns the global parameters of plasma with no information about its segmentation.

        Returns
        -------
        CylindricalPlasma.PlasmaMetrics
        """
        return self._plasma_metrics

    @property
    def grid_metrics(self):
        """
        Returns the params of plasma segmentation.

        Returns
        -------
        CylindricalPlasma.GridMetrics

        See also
        --------
        CylindricalPlasma.build_segmentation
        """
        return self._grid_metrics

    @property
    def n_segments(self):
        """
        Returns number of segments in segmentation.

        Returns
        -------
        int

        See also
        --------
        CylindricalPlasma.build_segmentation
        """
        return len(self.segments)

    @property
    def n_pol_rings(self):
        gm = self.grid_metrics
        return int(np.ceil(min(gm.n_r, gm.n_z) / 2))

    @property
    def lum(self):
        """
        Returns the luminosity vector of segments
        (by default all elements is 0 through no luminosity set).

        Returns
        -------
        np.ndarray

        See also
        --------
        CylindricalPlasma.lum_gradient
        """
        return np.array([s.lum for s in self.segments])

    def pol_rotation_state(self, phi=None):
        if phi is None:
            return self._rotation_status
        gm = self.grid_metrics
        if not 0 <= phi < gm.n_phi:
            raise ValueError("'phi' must satisfy '0 <= phi < gm.n_phi'.")
        return self._rotation_status[phi]

    def build_segmentation(self, n_r=1, n_phi=1, n_z=1):
        """
        Builds the segmentation with given segmentation numbers
        (all numbers must be positive).

        Parameters
        ----------
        n_r : int, optional
        n_phi : int, optional
        n_z : int, optional

        Notes
        -----
        If 'n_phi' = 1 then FullCircleSegment(s) will be generated, otherwise
        PhiBoundedSegment(s) will be generated.

        See also
        --------
        FullCircleSegment
        PhiBoundedSegment
        """
        if n_r <= 0 or n_phi <= 0 or n_z <= 0:
            raise ValueError('Numbers of segmentation must be positive.')

        pm = self._plasma_metrics
        d_r = (pm.r_max - pm.r_min) / n_r
        d_phi = 2.0 * pi / n_phi
        d_z = (pm.z_max - pm.z_min) / n_z
        self._grid_metrics = self.GridMetrics(n_r, n_phi, n_z, d_r, d_phi, d_z)
        self.segments.clear()

        z = pm.z_min
        if n_phi == 1:
            for i in range(n_z):
                r = pm.r_min
                for j in range(n_r):
                    segment = FullCircleSegment(r, r + d_r, z, z + d_z)
                    self.segments.append(segment)
                    r += d_r
                z += d_z
        else:
            for i in range(n_z):
                r = pm.r_min
                for j in range(n_r):
                    phi = self.phi_0
                    for k in range(n_phi):
                        segment = PhiBoundedSegment(r, r + d_r, phi, phi + d_phi, z, z + d_z)
                        self.segments.append(segment)
                        phi += d_phi
                    r += d_r
                z += d_z

        n_pol_rings = self.n_pol_rings
        self._rotation_status = [[0.0] * n_pol_rings for phi in range(n_phi)]
        # some empirical magic
        if n_r > n_z and n_r % 2 and n_z % 2:
            for phi in range(n_phi):
                self._rotation_status[phi][n_pol_rings - 1] = 0.5
        assert self.n_segments == n_r * n_phi * n_z

    def rotate(self, phi):
        """
        Rotates the plasma (anticlockwise looking from the top).
        If 'grid_metrics.n_phi' == 1, then does nothing but changes 'phi_0'.

        Parameters
        ----------
        phi : float
            Angle of rotation.

        See also
        --------
        CylindricalPlasma.phi_0
        """
        self._phi_0 = angle_normalize(self._phi_0 + phi)
        if self.grid_metrics.n_phi == 1:
            return
        for segment in self.segments:
            segment.rotate(phi)

    def lum_constant(self, lum=1.0):
        for segment in self.segments:
            segment.lum = lum

    def lum_piece_of_cake(self, lum=1.0, phi=0):
        """
        Generates 'piece of cake' luminosity for plasma segments.
        (Only the segments with 'phi' polar angle will have nonzero luminosity)

        Parameters
        ----------
        lum : float, optional
            The luminosity of segments.
        phi: int, optional
            Number of piece.

        See also
        --------
        CylindricalPlasma.lum
        """
        gm = self.grid_metrics
        if not 0 <= phi < gm.n_phi:
            raise ValueError("'phi' must satisfy '0 <= phi < gm.n_phi'.")
        self.lum_constant(0.0)
        for i in self.get_vertical_section(phi):
            self.segments[i].lum = lum

    def lum_toroidal_segment(self, lum_cor, lum_nuc):
        gm = self.grid_metrics
        r_edge = (gm.n_r - 1) // 2
        z_edge = (gm.n_z - 1) // 2
        self.lum_constant(0.0)
        for phi in range(gm.n_phi):
            for r in range(r_edge):
                for z in range(z_edge):
                    self.segments[self._index_by_r_phi_z(r, phi, z)].lum = lum_nuc
            for r in range(r_edge + 1):
                self.segments[self._index_by_r_phi_z(r, phi, z_edge)].lum = lum_cor
            for z in range(z_edge):
                self.segments[self._index_by_r_phi_z(r_edge, phi, z)].lum = lum_cor

    def lum_trapezoid(self, lum_cor, lum_nuc):
        gm = self.grid_metrics
        self.lum_constant(0.0)
        r_mid = gm.n_r // 2
        z_mid_1 = (gm.n_z - 1) // 2
        z_mid_2 = gm.n_z // 2
        for k in range(self.n_pol_rings):
            r = r_mid + k
            z1 = z_mid_1 - k
            z2 = z_mid_2 + k
            self.segments[self._index_by_r_phi_z(r, 0, z1)].lum = lum_cor
            self.segments[self._index_by_r_phi_z(r, 0, z2)].lum = lum_cor
            for z in range(z1 + 1, z2):
                self.segments[self._index_by_r_phi_z(r, 0, z)].lum = lum_nuc

    def lum_pol_rings(self, lum=1.0):
        gm = self.grid_metrics
        for phi in range(gm.n_phi):
            for i in range(self.n_pol_rings):
                for j in self.get_pol_ring(i, phi):
                    self.segments[j].lum = (i + 1) * lum

    def lum_gradient(self, lum_cor=0.0, lum_nuc=1.0):
        """
        Generates gradient luminosity for plasma segments.
        The luminosity will change from 'lum_cor' on the edges to 'lum_nuc' in the center.

        Parameters
        ----------
        lum_cor : float, optional
            The coronary luminosity (on the edges).
        lum_nuc : float, optional
            The nuclear luminosity (in the center).

        Notes
        -----
        The center is defined considering the cylinder as a torus with a rectangular cross section.
        So, the center will be a circle with z = '(z_min + z_max) / 2' and r = '(r_min + r_max) / 2'.

        See also
        --------
        CylindricalPlasma.lum
        """
        gm = self.grid_metrics
        n_r_div_2 = gm.n_r // 2
        n_r_mod_2 = gm.n_r % 2
        n_z_div_2 = gm.n_z // 2
        n_z_mod_2 = gm.n_z % 2
        dist_max = n_r_div_2 + n_r_mod_2 + n_z_div_2 + n_z_mod_2 - 2

        def nuc_dist(index, n_div_2, n_mod_2):
            if n_mod_2:
                return abs(index - n_div_2)
            if index < n_div_2:
                return n_div_2 - index - 1
            return index - n_div_2

        for i in range(self.n_segments):
            if dist_max == 0:
                dist = 0
            else:
                r, _, z = self._r_phi_z_by_index(i)
                r_dist = nuc_dist(r, n_div_2=n_r_div_2, n_mod_2=n_r_mod_2)
                z_dist = nuc_dist(z, n_div_2=n_z_div_2, n_mod_2=n_z_mod_2)
                dist = (r_dist + z_dist) / dist_max
            self.segments[i].lum = lum_cor * dist + lum_nuc * (1.0 - dist)

    def get_horizontal_section(self, z=0):
        gm = self.grid_metrics
        if not 0 <= z < gm.n_z:
            raise ValueError("'z' must satisfy '0 <= z < gm.n_z'")
        start_index = z * gm.n_phi * gm.n_r
        return range(start_index, start_index + gm.n_phi * gm.n_r)

    def get_vertical_section(self, phi=0):
        gm = self.grid_metrics
        if not 0 <= phi < gm.n_phi:
            raise ValueError("'phi' must satisfy '0 <= phi < gm.n_phi'.")
        for z in range(gm.n_z):
            for r in range(gm.n_r):
                yield self._index_by_r_phi_z(r, phi, z)

    def get_pol_ring(self, ring, phi=0):
        gm = self.grid_metrics
        if not 0 <= ring < self.n_pol_rings:
            raise ValueError("'ring' must satisfy '0 <= ring < self.n_pol_rings'.")
        if not 0 <= phi < gm.n_phi:
            raise ValueError("'phi' must satisfy '0 <= phi < gm.n_phi'.")
        if gm.n_r - 2 * ring == 1:
            yield from (self._index_by_r_phi_z(ring, phi, z) for z in range(ring, gm.n_z - ring))
            return
        if gm.n_z - 2 * ring == 1:
            yield from (self._index_by_r_phi_z(r, phi, ring) for r in range(ring, gm.n_r - ring))
            return

        for r in range(ring, gm.n_r - ring - 1):
            yield self._index_by_r_phi_z(r, phi, ring)
        for z in range(ring, gm.n_z - ring - 1):
            yield self._index_by_r_phi_z(gm.n_r - ring - 1, phi, z)
        for r in range(gm.n_r - ring - 1, ring, -1):
            yield self._index_by_r_phi_z(r, phi, gm.n_z - ring - 1)
        for z in range(gm.n_z - ring - 1, ring, -1):
            yield self._index_by_r_phi_z(ring, phi, z)

    def shift_pol(self, n_steps=1):
        gm = self.grid_metrics
        cycle_lgh = len(list(self.get_pol_ring(0)))

        for phi in range(gm.n_phi):
            state = self.pol_rotation_state(phi)
            for i in range(self.n_pol_rings):
                ring = list(self.get_pol_ring(i, phi))
                state[i] += n_steps * len(ring) / cycle_lgh
                if gm.n_r - 2 * i == 1 or gm.n_z - 2 * i == 1:
                    half_ring = len(ring) / 2.0
                    shift = int(round((state[i]) / half_ring))
                    state[i] -= half_ring * shift
                    if shift % 2:
                        self._ring_reverse(ring)
                else:
                    shift = int(round(state[i]))
                    state[i] -= shift
                    self._ring_rotate(ring, shift)

    def _index_by_r_phi_z(self, r, phi, z):
        gm = self.grid_metrics
        return phi + r * gm.n_phi + z * gm.n_phi * gm.n_r

    def _r_phi_z_by_index(self, index):
        gm = self.grid_metrics
        phi = index % gm.n_phi
        index //= gm.n_phi
        r = index % gm.n_r
        z = index // gm.n_r
        return r, phi, z

    def _ring_rotate(self, ring, shift):
        ring_segments = [self.segments[i] for i in ring]
        lum_arr = deque([s.lum for s in ring_segments])
        lum_arr.rotate(shift)
        for i in range(len(ring_segments)):
            ring_segments[i].lum = lum_arr[i]

    def _ring_reverse(self, ring):
        ring_segments = [self.segments[i] for i in ring]
        lum_arr = [s.lum for s in ring_segments]
        lum_arr.reverse()
        for i in range(len(ring_segments)):
            ring_segments[i].lum = lum_arr[i]
