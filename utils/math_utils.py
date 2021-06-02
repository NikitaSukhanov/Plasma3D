import functools

import numpy as np

EPSILON = 1e-10


def dist(p, q):
    return np.linalg.norm(p - q)


class Vector2D(np.ndarray):
    class _CoordinateDescriptor:
        def __init__(self, index):
            self._index = index

        def __get__(self, instance, owner):
            return instance[self._index]

        def __set__(self, instance, value):
            instance[self._index] = value

    x = _CoordinateDescriptor(0)
    y = _CoordinateDescriptor(1)

    def __new__(cls, x=0.0, y=0.0, **kwargs):
        kwargs.setdefault('dtype', float)
        return np.array([x, y], **kwargs).view(cls)

    @property
    def norm(self):
        return np.linalg.norm(self)

    @property
    def r(self):
        return _r_from_xy(self.x, self.y)

    @property
    def phi(self):
        return _phi_from_xy(self.x, self.y)

    def normalize(self):
        norm = self.norm
        if norm > EPSILON:
            self.__imul__(1.0 / norm)
        return self

    def dist(self, other):
        return dist(self, other)

    def rotate_2d(self, phi):
        sin, cos = np.sin(phi), np.cos(phi)
        x, y = self.x, self.y
        self.x = x * cos - y * sin
        self.y = x * sin + y * cos
        return self


class Vector3D(Vector2D):
    class _SliceDescriptor:
        def __get__(self, instance, owner):
            return instance[:2].view(Vector2D)

        def __set__(self, instance, value):
            instance[:2] = value

    xy = _SliceDescriptor()
    z = Vector2D._CoordinateDescriptor(2)

    def __new__(cls, x=0.0, y=0.0, z=0.0, **kwargs):
        kwargs.setdefault('dtype', float)
        return np.array([x, y, z], **kwargs).view(cls)

    def rotate_3d(self, e, phi):
        e = Vector3D(*e).normalize()
        sin, cos = np.sin(phi), np.cos(phi)
        inv_cos = 1.0 - cos
        x, y, z = e
        rotation_matrix = np.array([[cos + inv_cos * x ** 2, inv_cos * x * y - sin * z, inv_cos * x * z + sin * y],
                                    [inv_cos * y * x + sin * z, cos + inv_cos * y ** 2, inv_cos * y * z - sin * x],
                                    [inv_cos * x * z - sin * y, inv_cos * z * y + sin * x, cos + inv_cos * z ** 2]])
        self[:] = rotation_matrix.dot(self)
        return self

    @classmethod
    def from_2d(cls, xy, z=0.0):
        return cls(xy[0], xy[1], z)


@functools.lru_cache(maxsize=256)
def _r_from_xy(x, y):
    return np.sqrt(x ** 2 + y ** 2)


@functools.lru_cache(maxsize=256)
def _phi_from_xy(x, y):
    r = _r_from_xy(x, y)
    if abs(r) < EPSILON:
        return 0.0
    cos_phi = x / r
    sin_phi = y / r
    phi = np.arccos(cos_phi)
    if sin_phi < 0.0:
        phi = 2.0 * np.pi - phi
    return phi
