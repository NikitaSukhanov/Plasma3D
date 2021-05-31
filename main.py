import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from cylindrical_plasma import CylindricalPlasma, PlasmaDrawable
from detector import Detector, DetectorDrawable
from utils import iter_to_str
from math_utils import Vector3D, EPSILON, dist


def generate_file(plasma, detector, filename_prefix=''):
    """ Generates data-file for given plasma and detector. """
    # Ax = b
    A = detector.build_chord_matrix(plasma)
    x = plasma.solution
    b = detector.right_part(plasma)
    assert dist(A.dot(x), b) < EPSILON

    # generating full name of file
    filename = filename_prefix + '{}x{}.txt'.format(detector.n_pixels, plasma.n_segments)
    try:
        with open(filename, 'w') as file:
            # writing x, b and A into file, separated by one blank line
            file.write(iter_to_str(x) + '\n\n')
            file.write(iter_to_str(b) + '\n\n')
            for row in A:
                file.write(iter_to_str(row) + '\n')
            # writing geometry of plasma and detector after blank line
            file.write('\n' + plasma.__repr__() + '\n')
            file.write(detector.__repr__() + '\n')
    except IOError as error:
        print(error)


def example_datafile():
    """ Example of specifying geometry and generating data-file. """
    # plasma metrics:
    # Plasma is a cylinder ring given by 'r_min <= r <= r_max' and 'z_min <= z <= z_max'.
    r_min = 0.0
    r_max = 1.0
    z_min = -1.0
    z_max = 1.0
    # Plasma will be divided into 'n_r' parts by radius,
    # 'n_phi' parts by polar angle and 'n_z' parts by height.
    n_r = 6
    n_phi = 6
    n_z = 6
    # Plasma will be rotated by 'phi_0' angle around 'Z' axis comparing to its default state.
    phi_0 = 3.6411293856408307  # (~ pi + 1/2)
    # The luminosity of plasma will be changing gradiently,
    # for 'lum_nuc' at the center to 'lum_cor' on the edges.
    lum_cor = 0.0
    lum_nuc = 1.0

    # detector metrics
    center = -1.5, -1.5, 0.0  # center of detector's matrix
    aperture = -1.1, -1.0, 0.0  # position of detector's aperture
    # Detector's matrix is the rectangle with sides of size 'height' and 'width'.
    height = 0.6
    width = 0.6
    # The detector's matrix could be rotated around its normal by 'angle'.
    angle = 0.0
    # Detector will have 'rows' lines of pixels, each line contains 'cols' pixels.
    rows = 16
    cols = 16

    # creating plasma with given metrics
    plasma = CylindricalPlasma(r_min=r_min, r_max=r_max, z_min=z_min, z_max=z_max)
    # dividing plasma into segments (by default plasma is one big segment)
    plasma.build_segmentation(n_r=n_r, n_phi=n_phi, n_z=n_z)
    # rotating plasma (and all its segments) by given angle around 'Z' axis
    plasma.rotate(phi_0)
    # assigning gradient luminosity to each segment
    plasma.lum_gradient(lum_cor=lum_cor, lum_nuc=lum_nuc)

    # creating detector with given metrics
    detector = Detector(center=center, aperture=aperture, height=height, width=width, angle=angle)
    # setting given amount of pixels on detector (by default detector has one pixel at the center)
    detector.set_pixels(rows=rows, cols=cols)
    # generating data-file with specified geometry
    generate_file(plasma=plasma, detector=detector, filename_prefix='examples/gradient_')

    # assigning 'piece of cake' luminosity to each segment
    plasma.lum_piece_of_cake(lum=lum_nuc)
    # generating data-file with specified geometry
    generate_file(plasma=plasma, detector=detector, filename_prefix='examples/poc_')


def example_plot():
    """
    Example of drawing geometry setup
    with several detectors in different positions and orientations.
    """
    # creating plasma and 3 different detectors
    plasma = PlasmaDrawable(r_min=0.0, r_max=1.0, z_min=-1.0, z_max=1.0)
    plasma.build_segmentation(n_r=2, n_phi=4, n_z=2)
    d1 = DetectorDrawable(center=(-1.5, -1.5, 0.0), aperture=(-1.1, -1.0, 0.0), height=0.6, width=0.6)
    d1.set_pixels(rows=3, cols=3)
    d2 = DetectorDrawable(center=(0.0, 0.0, 2.5), aperture=(0.0, 0.0, 1.5), height=0.3, width=0.4)
    d2.set_pixels(rows=2, cols=2)
    d3 = DetectorDrawable(center=(1.5, -1.2, 0.1), aperture=(1.4, -1.1, 0.07), height=0.1, width=0.2, angle=np.pi / 4)
    d3.set_pixels(rows=4, cols=1)

    # setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_title('Example plot')

    # plot plasma
    plasma.plot(ax, color='red')
    for detector, color in zip((d1, d2, d3), ('green', 'blue', 'black')):
        # plot each detector
        detector.plot(ax, lines_length=detector.lines_length_calculate(plasma), color=color)
        # plot intersection points
        for segment in plasma.segments:
            for pixel in detector.pixels:
                points = segment.intersection_points(pixel.pos, pixel.dir)
                for _, point in points:
                    ax.scatter(*point, color=color)
    plt.show()


example_datafile()
example_plot()
