import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from plasma_lib.cylindrical_plasma import CylindricalPlasma
from plasma_lib.plasma_drawable import PlasmaDrawable
from plasma_lib.detector import Detector
from plasma_lib.detector_drawable import DetectorDrawable
from utils.math_utils import Vector3D, EPSILON, dist


def matrix_phi_sum(matrix, n_phi):
    m, n = matrix.shape
    if n % n_phi:
        raise ValueError('matrix.shape[1] % n_phi must be 0')
    n //= n_phi
    new_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            new_matrix[i][j] = sum(matrix[i][j * n_phi: (j + 1) * n_phi])
    return new_matrix


def matrix_sum_test(draw=True):
    center = (1.4, 1.5, 1.6)
    aperture = (1.1, 1.12, 1.13)
    n_r = 7
    n_phi = 8
    n_z = 9

    plasma_circle = PlasmaDrawable()
    plasma_separated = PlasmaDrawable()
    plasma_circle.build_segmentation(n_r=n_r, n_phi=1, n_z=n_z)
    plasma_separated.build_segmentation(n_r=n_r, n_phi=n_phi, n_z=n_z)
    detector = Detector(center=center, apertures=aperture, height=0.4, width=0.5, angle=0.1)

    matrix_circle = detector.build_chord_matrix(plasma_circle)
    matrix_separated = detector.build_chord_matrix(plasma_separated)
    matrix_summed = matrix_phi_sum(matrix_separated, n_phi)

    m1, n1 = matrix_circle.shape
    m2, n2 = matrix_separated.shape
    m3, n3 = matrix_summed.shape
    assert m1 == m2 and m1 == m3
    assert n1 == n3 and n2 == n1 * n_phi

    residual = matrix_circle - matrix_summed
    res_norm = np.linalg.norm(residual)
    for i in range(m1):
        for j in range(n1):
            if residual[i][j] > EPSILON:
                print('Elements summation test failed.')
                print('||residual|| = {}'.format(res_norm))
                if not draw:
                    return False
                pixel = detector.pixels[i]

                # left plot
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax1.grid(False)
                ax1.set_title('Segment{}, pixel{} (len={})'.format(j, i, matrix_circle[i][j]))
                plasma_circle.plot(ax1)
                circle_segment = plasma_circle.segments[j]
                lgh = circle_segment.intersection_length(pixel.pos, pixel.dir)
                print('Circle segment{}: {}'.format(j, circle_segment.__repr__()))
                print('Circle segment{}: Intersection len = {}'.format(j, lgh))
                for _, point in circle_segment.intersection_points(pixel.pos, pixel.dir):
                    ax1.scatter(*point, color='blue')

                # right plot
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                ax2.grid(False)
                ax2.set_title('Segments{}-{}, pixel{} (len={})'.
                              format(j * n_phi, (j + 1) * n_phi, i, matrix_summed[i][j]))
                plasma_separated.plot(ax2)
                for k in range(n_phi):
                    index = j * n_phi + k
                    segment = plasma_separated.segments[index]
                    lgh = segment.intersection_length(pixel.pos, pixel.dir)
                    print('Bounded segment{}: {}'.format(index, segment.__repr__()))
                    print('Bounded segment{}: Intersection len = {}'.format(index, lgh))
                    for _, point in segment.intersection_points(pixel.pos, pixel.dir):
                        ax2.scatter(*point, color='blue')
                plt.show()
                return False
    return True


def rotation_test():
    center = (1.4, 1.5, 1.6)
    aperture = (1.1, 1.12, 1.13)
    n_r = 7
    n_phi = 8
    n_z = 9

    p1 = PlasmaDrawable()
    p1.build_segmentation(n_r=n_r, n_phi=n_phi, n_z=n_z)
    p1.lum_trapezoid(1, 0.5)
    p2 = PlasmaDrawable()
    p2.build_segmentation(n_r=n_r, n_phi=n_phi, n_z=n_z)
    p2.lum_trapezoid(1, 0.5)

    lum_initial = p1.lum
    state_initial = np.array(p1.pol_rotation_state())
    cycle_lgh = len(list(p1.get_pol_ring(0)))

    for shift in (-4, -1, 5, 34, 1):
        for i in range(abs(shift)):
            p1.shift_pol(np.sign(shift))
        p2.shift_pol(shift)
        if dist(p1.lum, p2.lum) > EPSILON:
            return False
        if dist(np.array(p1.pol_rotation_state()), np.array(p2.pol_rotation_state())) > EPSILON:
            return False

        p1.shift_pol(-shift)
        p2.shift_pol(2 * cycle_lgh - shift)
        if dist(p1.lum, p2.lum) > EPSILON:
            return False
        if dist(np.array(p1.pol_rotation_state()), np.array(p2.pol_rotation_state())) > EPSILON:
            return False
        if dist(p1.lum, lum_initial) > EPSILON:
            return False
        if dist(np.array(p1.pol_rotation_state()), state_initial) > EPSILON:
            return False
        return True


assert matrix_sum_test()
assert rotation_test()
