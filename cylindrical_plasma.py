from operator import itemgetter
from collections import namedtuple

import numpy as np

from math_utils import Vector2D, Vector3D, EPSILON
from utils import documentation_inheritance


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
    return angle % (2.0 * np.pi)


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
        Generator of solution (if there are no solutions generator will be empty).
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
        self._r_min = r_min
        self._r_max = r_max
        self._z_min = z_min
        self._z_max = z_max
        self.lum = lum

    def __repr__(self):
        state = 'r_min={}, r_max={}, z_min={}, z_max={}'
        return state.format(self._r_min, self._r_max, self._z_min, self._z_max)

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
        for z in (self._z_min, self._z_max):
            points.extend(self._z_surface_intersection(z, p, e))
        points.extend(self._r_surface_intersection(self._r_max, p, e))
        if self._r_min > EPSILON:
            points.extend(self._r_surface_intersection(self._r_min, p, e))
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
            if self._z_min <= q.z <= self._z_max:
                yield t, q

    def _z_surface_intersection(self, z, p, e):
        if abs(e.z) < EPSILON:
            return
        t = (z - p.z) / e.z
        q = p + t * e
        if self._r_min <= q.r <= self._r_max:
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
        self._phi_min = phi_min
        self._phi_max = phi_max
        self.rotate(0.0)

    def __repr__(self):
        state = 'r_min={}, r_max={}, phi_min={}, phi_max={}, z_min={}, z_max={}'
        return state.format(self._r_min, self._r_max, self._phi_min, self._phi_max, self._z_min, self._z_max)

    def rotate(self, phi):
        """
        Rotates the segment (anticlockwise looking from the top).

        Parameters
        ----------
        phi : float
            Angle of rotation.
        """
        phi = angle_normalize(phi)
        self._phi_min += phi
        self._phi_max += phi
        if self._phi_min >= 2.0 * np.pi:
            self._phi_min = angle_normalize(self._phi_min)
            self._phi_max = angle_normalize(self._phi_max)
        assert self._phi_min <= self._phi_max

    @documentation_inheritance(FullCircleSegment.intersection_points)
    def intersection_points(self, p, e):
        points = []
        for z in (self._z_min, self._z_max):
            points.extend(self._z_surface_intersection(z, p, e))
        for phi in (self._phi_min, self._phi_max):
            points.extend(self._phi_surface_intersection(phi, p, e))
        for r in (self._r_min, self._r_max):
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
        if EPSILON < abs(phi - q.phi) < 2.0 * np.pi - EPSILON:
            # opposite half-plane
            return
        if self._z_min <= q.z <= self._z_max and self._r_min <= q.r <= self._r_max:
            yield t, q

    def _phi_bounds_check(self, phi):
        phi = angle_normalize(phi)
        if self._phi_max <= 2.0 * np.pi:
            return self._phi_min <= phi <= self._phi_max
        return self._phi_min <= phi <= 2.0 * np.pi or 0.0 <= phi <= angle_normalize(self._phi_max)


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
        - solution
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
        self.build_segmentation(n_r=1, n_phi=1, n_z=1)

    def __repr__(self):
        state = 'n_segments={}, phi0={}\n'.format(self.n_segments, self.phi_0)
        state += self.plasma_metrics.__repr__() + '\n'
        state += self.grid_metrics.__repr__()
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
    def solution(self):
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
        d_phi = 2.0 * np.pi / n_phi
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
            return

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

    def lum_gradient(self, lum_cor, lum_nuc):
        """
        Generates gradient luminosity for plasma segments.
        The luminosity will change from 'lum_cor' on the edges to 'lum_nuc' in the center.

        Parameters
        ----------
        lum_cor : float
            The coronary luminosity (on the edges).
        lum_nuc : float
            The nuclear luminosity (in the center).

        Notes
        -----
        The center is defined considering the cylinder as a torus with a rectangular cross section.
        So, the center will be a circle with z = '(z_min + z_max) / 2' and r = '(r_min + r_max) / 2'.

        See also
        --------
        CylindricalPlasma.solution
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
                index = i // gm.n_phi
                r = index % gm.n_r
                r = nuc_dist(r, n_div_2=n_r_div_2, n_mod_2=n_r_mod_2)
                index //= gm.n_r
                z = index % gm.n_z
                z = nuc_dist(z, n_div_2=n_z_div_2, n_mod_2=n_z_mod_2)
                dist = (r + z) / dist_max
            self.segments[i].lum = lum_cor * dist + lum_nuc * (1.0 - dist)


class PlasmaDrawable(CylindricalPlasma):
    """ A drawable wrapper over the CylindricalPlasma. """

    @documentation_inheritance(CylindricalPlasma.__init__)
    def __init__(self, r_min=0.0, r_max=1.0, z_min=-1.0, z_max=1.0):
        super().__init__(r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max)

    def plot(self, ax, **kwargs):
        """
        Draws the plasma.

        Parameters
        ----------
        ax :
            Axes3D object from matplotlib.
        kwargs :
            Keyword arguments which will be passed to all 'plot' and 'plot_surface' methods.
        """
        kwargs.setdefault('color', 'r')
        self._plot_outer_surface(ax, **kwargs)
        self._plot_rz_separation(ax, **kwargs)
        self._plot_phi_separation(ax, **kwargs)

    def _plot_outer_surface(self, ax, **kwargs):
        pm = self._plasma_metrics
        kwargs.setdefault('alpha', 0.2)
        num = 50

        # drawing outer and inner vertical cylinders
        z = np.linspace(pm.z_min, pm.z_max, num)
        phi = np.linspace(0.0, 2.0 * np.pi, num)
        z_grid, phi_grid = np.meshgrid(z, phi)
        cos_phi = np.cos(phi_grid)
        sin_phi = np.sin(phi_grid)
        x1_grid = pm.r_max * cos_phi
        y1_grid = pm.r_max * sin_phi
        x2_grid = pm.r_min * cos_phi
        y2_grid = pm.r_min * sin_phi
        ax.plot_surface(x1_grid, y1_grid, z_grid, **kwargs)
        ax.plot_surface(x2_grid, y2_grid, z_grid, **kwargs)

        # drawing lower and upper cups
        r = np.linspace(pm.r_min, pm.r_max, num)
        r_grid, phi_grid = np.meshgrid(r, phi)
        x_grid = r_grid * np.cos(phi_grid)
        y_grid = r_grid * np.sin(phi_grid)
        z1_grid = np.ones_like(x_grid) * pm.z_min
        z2_grid = np.ones_like(x_grid) * pm.z_max

        ax.plot_surface(x_grid, y_grid, z1_grid, **kwargs)
        ax.plot_surface(x_grid, y_grid, z2_grid, **kwargs)

    def _plot_rz_separation(self, ax, **kwargs):
        pm = self._plasma_metrics
        gm = self.grid_metrics
        num = 50
        phi = np.linspace(0.0, 2.0 * np.pi, num)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        r = pm.r_min
        for i in range(gm.n_r + 1):
            x = r * cos_phi
            y = r * sin_phi
            z = pm.z_min
            for j in range(gm.n_z + 1):
                ax.plot(x, y, z, **kwargs)
                z += gm.d_z
            r += gm.d_r

    def _plot_phi_separation(self, ax, **kwargs):
        pm = self._plasma_metrics
        gm = self.grid_metrics
        if gm.n_phi == 1:
            return

        phi = self.phi_0
        for i in range(gm.n_phi):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            phi += gm.d_phi

            # draw horizontal lines
            x1 = pm.r_min * cos_phi
            y1 = pm.r_min * sin_phi
            x2 = pm.r_max * cos_phi
            y2 = pm.r_max * sin_phi
            z = pm.z_min
            for j in range(gm.n_z + 1):
                ax.plot([x1, x2], [y1, y2], z, **kwargs)
                z += gm.d_z

            # draw vertical lines
            z = [pm.z_min, pm.z_max]
            r = pm.r_min
            for j in range(gm.n_r + 1):
                x = r * cos_phi
                y = r * sin_phi
                ax.plot([x, x], [y, y], z, **kwargs)
                r += gm.d_r
