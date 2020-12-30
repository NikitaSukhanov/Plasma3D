from collections import namedtuple

import numpy as np

from math_utils import Vector3D, EPSILON


class Pixel:
    def __init__(self, pos, aperture):
        self._pos = pos.copy()
        self._e = (aperture - pos).normalize()

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._e


class Detector:
    DetectorMetrics = namedtuple('DetectorMetrics', ['aperture', 'center', 'corner',
                                                     'normal', 'tangent', 'top',
                                                     'height', 'width', 'angle'])

    def __init__(self, center, aperture, height=1.0, width=1.0, angle=0.0):
        aperture = Vector3D(*aperture)
        center = Vector3D(*center)
        normal = (aperture - center).normalize()
        if normal.norm <= EPSILON:
            raise ValueError("'center' and 'aperture' could not coincide")
        if height < 0.0 or width < 0.0:
            raise ValueError("'height' and 'width' could not be negative")

        if abs(normal.x) >= EPSILON or abs(normal.y) >= EPSILON:
            tangent = Vector3D(-normal.y, normal.x, 0.0).normalize().rotate_3d(normal, angle)
            top = Vector3D(*np.cross(normal, tangent)).normalize()

        else:
            normal = Vector3D(0.0, 0.0, np.sign(normal.z))
            tangent = Vector3D(1.0, 0.0, 0.0).rotate_2d(angle)
            top = Vector3D(0.0, 1.0, 0.0).rotate_2d(angle)

        corner = center - tangent * width / 2.0 - top * height / 2.0
        self._detector_metrics = Detector.DetectorMetrics(aperture=aperture, center=center, corner=corner,
                                                          normal=normal, tangent=tangent, top=top,
                                                          height=height, width=width, angle=angle)
        self.pixels = []
        self._rows = 0
        self._cols = 0
        self.set_pixels(rows=1, cols=1)

    def __repr__(self):
        state = 'n_pixels={}, rows={}, cols={}\n'.format(self.n_pixels, self._rows, self._cols)
        state += self.detector_metrics.__repr__()
        return state

    @property
    def detector_metrics(self):
        return self._detector_metrics

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def n_pixels(self):
        return len(self.pixels)

    def set_pixels(self, rows=1, cols=1):
        if rows <= 0 or cols <= 0:
            raise ValueError('Numbers of rows and columns must be positive.')

        dm = self.detector_metrics
        self._rows = rows
        self._cols = cols
        self.pixels.clear()
        start = dm.center.copy()
        d_h = 0
        d_w = 0

        if rows > 1:
            start -= dm.top * dm.height / 2.0
            d_h = dm.height / (rows - 1)
        if cols > 1:
            start -= dm.tangent * dm.width / 2.0
            d_w = dm.width / (cols - 1)

        for i in range(rows):
            pos = start + i * d_h * dm.top
            for j in range(cols):
                self.pixels.append(Pixel(pos, dm.aperture))
                pos += d_w * dm.tangent
        assert self.n_pixels == self.cols * self.rows

    def build_chord_matrix(self, plasma):
        n = plasma.n_segments
        m = self.n_pixels
        matrix = np.zeros((m, n))
        for i in range(m):
            p, e = self.pixels[i].pos, self.pixels[i].dir
            for j in range(n):
                matrix[i][j] = plasma.segments[j].intersection_length(p, e)
        return matrix

    def right_part(self, plasma):
        m = self.n_pixels
        res = np.zeros(m)
        for i in range(m):
            p, e = self.pixels[i].pos, self.pixels[i].dir
            for segment in plasma.segments:
                res[i] += segment.lum * segment.intersection_length(p, e)
        return res


class DetectorDrawable(Detector):
    def __init__(self, center, aperture, height=1.0, width=1.0, angle=0.0):
        super().__init__(center=center, aperture=aperture, height=height, width=width, angle=angle)

    def plot(self, ax, lines_length=None, **kwargs):
        kwargs.setdefault('color', 'b')
        self._plot_plane(ax, **kwargs)
        self._plot_borders(ax, **kwargs)
        self._plot_pixels(ax, **kwargs)
        self._plot_aperture(ax, **kwargs)
        if lines_length:
            self._plot_lines(ax, lines_length, **kwargs)

    def lines_length_calculate(self, plasma):
        pm = plasma.plasma_metrics
        plasma_center = Vector3D(0.0, 0.0, (pm.z_min + pm.z_max) / 2.0)
        plasma_radius = plasma_center.dist((pm.r_max, 0.0, pm.z_max))
        corners = self._corners_calculate()
        max_dist = max(plasma_center.dist(corner) for corner in corners)
        return max_dist + plasma_radius

    def _plot_plane(self, ax, **kwargs):
        dm = self._detector_metrics
        num = 2
        h_grid, w_grid = np.meshgrid(np.linspace(0, dm.height, num), np.linspace(0, dm.width, num))
        grid_gen = (dm.corner[i] + h_grid * dm.top[i] + w_grid * dm.tangent[i] for i in range(3))
        kwargs.setdefault('alpha', 0.2)
        ax.plot_surface(*grid_gen, **kwargs)

    def _plot_borders(self, ax, **kwargs):
        corners = self._corners_calculate()
        ax.plot(*corners.T, **kwargs)

    def _plot_pixels(self, ax, **kwargs):
        kwargs.setdefault('marker', 's')
        for pixel in self.pixels:
            ax.scatter(*pixel.pos, **kwargs)

    def _plot_aperture(self, ax, **kwargs):
        dm = self._detector_metrics
        kwargs.setdefault('facecolors', [0.0, 0.0, 0.0, 0.0])
        ax.scatter(*dm.aperture, **kwargs)

    def _plot_lines(self, ax, lines_length, **kwargs):
        for pixel in self.pixels:
            p1 = pixel.pos
            p2 = p1 + pixel.dir * lines_length
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], **kwargs)

    def _corners_calculate(self):
        dm = self._detector_metrics
        return np.array([dm.corner,
                         dm.corner + dm.tangent * dm.width,
                         dm.corner + dm.tangent * dm.width + dm.top * dm.height,
                         dm.corner + dm.top * dm.height,
                         dm.corner])
