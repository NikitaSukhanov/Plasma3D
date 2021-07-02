from collections import namedtuple

import numpy as np

from utils.math_utils import Vector3D, EPSILON


class Pixel:
    """ Single pixel of detector, has its position and direction to aperture"""

    def __init__(self, pos, aperture):
        """
        Parameters
        ----------
        pos : Vector3D
            Position of pixel.
        aperture : Vector3D
            Position of aperture.
        """
        self._pos = Vector3D(*pos)
        self._e = (aperture - pos).normalize()

    @property
    def pos(self):
        """
        Returns
        -------
        Vector3D
            Position of pixel.
        """
        return self._pos

    @property
    def dir(self):
        """
        Returns
        -------
        Vector3D
            Direction to aperture.
        """
        return self._e


class Detector:
    """
    Detector is a rectangular plate of pixels with the aperture point outside the plate.

    Attributes:
    -----------
        - detector_metrics
        - rows
        - cols
        - n_pixels
    """
    DetectorMetrics = namedtuple('DetectorMetrics', ['aperture', 'center', 'corner',
                                                     'normal', 'tangent', 'top',
                                                     'height', 'width', 'angle'])

    def __init__(self, center, aperture, height=1.0, width=1.0, angle=0.0):
        """
        The position and orientation of the plate is determined by following:
            - Center of plate is given by 'center'.
            - Plate is always oriented such way that it's normal vector points
              from the center to aperture.
            - After first two steps the only remaining degree of freedom is the
              angle of rotation of plate around it's normal, which is given by 'angle'.
            - By default ('angle = 0') the 'width' side of the plate is aligned with XY plane.
              If the plate is horizontal then by default 'width' is aligned with X axis and
              'height' is aligned with Y axis.

        Parameters
        ----------
        center : iterable
            Center of rectangular plate (detector matrix).
        aperture : iterable
            Position of aperture.
        height : float, optional
            Height of rectangular plate.
        width : float, optional
            Height of rectangular plate.
        angle : float, optional
            Angle of rotation of plate around its normal.
        """
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
        """
        Returns the parameters of detector.
        Orientation of plate is determined by local basis of 3 orthogonal vectors:

        - 'normal' - vector which is orthogonal to plate
        - 'tangent' - vector aligned with 'width' of plate
        - 'top' - vector aligned with 'height' of plate

        Returns
        -------
        Detector.DetectorMetrics
        """
        return self._detector_metrics

    @property
    def rows(self):
        """
        Returns number of rows of pixels.

        Returns
        -------
        int
        """
        return self._rows

    @property
    def cols(self):
        """
        Returns number of pixels in each row.

        Returns
        -------
        int
        """
        return self._cols

    @property
    def n_pixels(self):
        """
        Returns total number of pixels on detector plate.

        Returns
        -------
        int
        """
        return len(self.pixels)

    def set_pixels(self, rows=1, cols=1):
        """
        Specifies the number of pixels.

        Parameters
        ----------
        rows : int, optional
        cols : int, optional
        """
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
        """
        Builds the chord matrix by given plasma model, '(i, j)' element of matrix
        is the length of intersection of ray from 'i' pixel with 'j' segment.

        Parameters
        ----------
        plasma : Any
            Plasma must contain list 'plasma.segments' and int field 'plasma.n_segments'.
            Each segment in list must implement 'intersection_length(p, e)' method.

        Returns
        -------
        np.ndarray
        """
        n = plasma.n_segments
        m = self.n_pixels
        matrix = np.zeros((m, n))
        for i in range(m):
            p, e = self.pixels[i].pos, self.pixels[i].dir
            for j in range(n):
                matrix[i][j] = plasma.segments[j].intersection_length(p, e)
        return matrix

    def right_part(self, plasma):
        """
        Calculates the right-part vector of detector pixels' measurements.
        Parameters
        ----------
        plasma : Any
            Plasma must contain iterable 'plasma.segments'.
            Each segment must have 'segment.lum' float field
            and implement 'intersection_length(p, e)' method.

        Returns
        -------
        np.ndarray

        Notes
        -----
        The luminosity of segments must be specified before calling this.
        The result must be equal to 'detector.build_chord_matrix(plasma) @ plasma.solution'.
        """
        m = self.n_pixels
        res = np.zeros(m)
        for i in range(m):
            p, e = self.pixels[i].pos, self.pixels[i].dir
            for segment in plasma.segments:
                res[i] += segment.lum * segment.intersection_length(p, e)
        return res

    def _corners_calculate(self):
        dm = self._detector_metrics
        return np.array([dm.corner,
                         dm.corner + dm.tangent * dm.width,
                         dm.corner + dm.tangent * dm.width + dm.top * dm.height,
                         dm.corner + dm.top * dm.height,
                         dm.corner])
