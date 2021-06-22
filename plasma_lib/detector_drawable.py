from plasma_lib.detector import *


class DetectorDrawable(Detector):
    """ A drawable wrapper over the Detector. """

    @documentation_inheritance(Detector.__init__)
    def __init__(self, center, aperture, height=1.0, width=1.0, angle=0.0):
        super().__init__(center=center, aperture=aperture, height=height, width=width, angle=angle)

    def plot(self, ax, lines_length=None, **kwargs):
        """
        Draws the detector.

        Parameters
        ----------
        ax :
            Axes3D object from matplotlib.
        lines_length : float, optional
            The length of rays to draw from each pixel of detector.
            If None (by default) or 0 no rays will be drawn.
        kwargs :
             Keyword arguments which will be passed to all 'plot',
             'plot_surface' and 'scatter' methods.

        See also
        --------
        DetectorDrawable.lines_length_calculate
        """
        kwargs.setdefault('color', 'b')
        self._plot_plane(ax, **kwargs)
        self._plot_borders(ax, **kwargs)
        self._plot_pixels(ax, **kwargs)
        self._plot_aperture(ax, **kwargs)
        if lines_length:
            self._plot_lines(ax, lines_length, **kwargs)

    def plot_polar_projection(self, ax_polar, lines_length=None, **kwargs):
        """
        Draws the horizontal projection of detector.
        Requires polar axes !!!

        Parameters
        ----------
        ax_polar :
             PolarAxesSubplot object from matplotlib.
        lines_length : float, optional
            The length of rays to draw from each pixel of detector.
            If None (by default) or 0 no rays will be drawn.
        kwargs :
            Keyword arguments which will be passed to all 'plot',
            'plot_surface' and 'scatter' methods.

        See also
        --------
        DetectorDrawable.lines_length_calculate
        """
        dm = self.detector_metrics
        corners = self._corners_calculate()
        c1, c2 = Vector3D(*corners[0]), Vector3D(*corners[2])
        kwargs.setdefault('color', 'blue')
        ax_polar.plot([c1.phi, c2.phi], [c1.r, c2.r], **kwargs)
        if lines_length:
            for c in c1, c2:
                p = c + lines_length * (dm.aperture - c).normalize()
                ax_polar.plot([c.phi, p.phi], [c.r, p.r], **kwargs)
        kwargs.setdefault('facecolors', [0.0, 0.0, 0.0, 0.0])
        kwargs.setdefault('edgecolors', kwargs['color'])
        ax_polar.scatter(dm.aperture.phi, dm.aperture.r, **kwargs)

    def plot_right_part(self, ax, right_part, **kwargs):
        """
        Draws detector pixels' measurements in 2d pixel-plot.

        Parameters
        ----------
        ax :
            AxesSubplot object from matplotlib.
        right_part :  np.ndarray
            Vector of detector pixels' measurements.
        kwargs :
              Keyword arguments which will be passed 'pcolormesh' method.

        See also
        --------
        DetectorDrawable.right_part
        """
        if len(right_part) != self.n_pixels:
            raise ValueError("'right_part' must satisfy 'len(right_part) == self.n_pixels'")
        matrix = right_part.reshape((self.rows, self.cols))
        ax.pcolormesh(matrix, **kwargs)

    def lines_length_calculate(self, plasma):
        """
        Calculates the 'lines_length' argument for 'plot' method.
        The length is not less then distance from any detector pixel
        to any point of cylindrical plasma.

        Parameters
        ----------
        plasma : Any
            Plasma must have 'plasma.plasma_metrics' field,
            which has 'z_min', 'z_max' and 'r_max' float fields.

        Returns
        -------
        float
        """
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
