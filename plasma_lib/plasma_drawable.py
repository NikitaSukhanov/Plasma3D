import matplotlib.patches as patches

from plasma_lib.cylindrical_plasma import *


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

    def plot_horizontal_section(self, ax_polar, z_number=0, phi_sep=True, **kwargs):
        """
        Draws the horizontal section (on given level) of plasma.
        Requires polar axes !!!

        Parameters
        ----------
        ax_polar :
            PolarAxesSubplot object from matplotlib.
        z_number : int, optional
            Number of horizontal section
        phi_sep : bool, optional
            Flag, indicating whether to draw phi separation.
        kwargs :
            Keyword arguments which will be passed to all 'plot' and 'fill_between' methods.
        """
        pm = self._plasma_metrics
        gm = self.grid_metrics
        if not 0 <= z_number < gm.n_z:
            raise ValueError("'z_number' must satisfy '0 <= z_number < gm.n_z'")
        kwargs.setdefault('color', np.array([1.0, 0.0, 0.0]))
        kwargs.setdefault('linewidth', 1.0)
        num = 50

        # draw r separation (circles)
        ones = np.ones(num)
        phi = np.linspace(0.0, 2.0 * np.pi, num)
        r = pm.r_min
        for i in range(gm.n_r + 1):
            ax_polar.plot(phi, r * ones, **kwargs)
            r += gm.d_r

        # draw phi separation (lines)
        if gm.n_phi > 1 and phi_sep:
            phi = self.phi_0
            for i in range(gm.n_phi):
                ax_polar.plot([phi, phi], [pm.r_min, pm.r_max], **kwargs)
                phi += gm.d_phi

        # fill segments according to their luminance
        kwargs['linewidth'] = 0.0
        lum_max = max(self.lum)
        for i in self.get_horizontal_section(z_number):
            kwargs['alpha'] = self.segments[i].lum / (lum_max + 0.5)
            self.plot_segment_horizontal(ax_polar, i, **kwargs)

    def plot_vertical_section(self, ax, phi_number=0, grid_sep=True, **kwargs):
        pm = self._plasma_metrics
        gm = self.grid_metrics
        if not 0 <= phi_number < gm.n_phi:
            raise ValueError("'phi_number' must satisfy '0 <= phi_number < gm.n_phi'")
        kwargs.setdefault('color', np.array([1.0, 0.0, 0.0]))
        kwargs.setdefault('linewidth', 1.0)

        # draw full grid
        if grid_sep:
            z = pm.z_min
            for i in range(gm.n_z + 1):
                ax.plot([pm.r_min, pm.r_max], [z, z], **kwargs)
                z += gm.d_z
            r = pm.r_min
            for i in range(gm.n_r + 1):
                ax.plot([r, r], [pm.z_min, pm.z_max], **kwargs)
                r += gm.d_r
        # else draw only outer rectangle
        else:
            rect = patches.Rectangle((pm.r_min, pm.z_min), pm.r_max - pm.r_min, pm.z_max - pm.z_min,
                                     linewidth=kwargs['linewidth'],
                                     edgecolor=kwargs['color'],
                                     facecolor='none')
            ax.add_patch(rect)

        # fill segments according to their luminance
        kwargs['linewidth'] = 0.0
        lum_max = max(self.lum)
        for i in self.get_vertical_section(phi_number):
            kwargs['alpha'] = self.segments[i].lum / (lum_max + 0.5)
            self.plot_segment_vertical(ax, i, **kwargs)

    def plot_segment_horizontal(self, ax_polar, segment_number, **kwargs):
        """
        Draws the horizontal section of plasma segment.
        Requires polar axes !!!

        Parameters
        ----------
        ax_polar :
            PolarAxesSubplot object from matplotlib.
        segment_number : int
            Number of segment.
        kwargs :
            Keyword arguments which will be passed 'fill_between' method.
        """
        if not 0 <= segment_number < self.n_segments:
            raise ValueError("'segment_number' must satisfy '0 <= segment_number < self.n_segments'")
        segment = self.segments[segment_number]
        gm = self.grid_metrics
        num = 50
        if gm.n_phi == 1:
            assert segment.__class__ is FullCircleSegment
            phi = np.linspace(0.0, 2.0 * np.pi + EPSILON, num)
        else:
            assert segment.__class__ is PhiBoundedSegment
            phi = np.linspace(segment.phi_min, segment.phi_max, num)
        ax_polar.fill_between(phi, segment.r_min, segment.r_max, **kwargs)

    def plot_segment_vertical(self, ax, segment_number, **kwargs):
        """
        Draws the vertical section of plasma segment.

        Parameters
        ----------
        ax :
            AxesSubplot object from matplotlib.
        segment_number : int
            Number of segment.
        kwargs :
            Keyword arguments which will be passed 'fill_between' method.
        """
        if not 0 <= segment_number < self.n_segments:
            raise ValueError("'segment_number' must satisfy '0 <= segment_number < self.n_segments'")
        segment = self.segments[segment_number]
        num = 2
        r = np.linspace(segment.r_min, segment.r_max, num)
        ax.fill_between(r, segment.z_min, segment.z_max, **kwargs)

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
