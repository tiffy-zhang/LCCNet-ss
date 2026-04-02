import numpy as np
from scipy.spatial.transform import Rotation


def _as_numpy(values):
    try:
        import torch
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(values, dtype=np.float64)


class Vector:
    def __init__(self, values):
        arr = _as_numpy(values).reshape(-1)
        if arr.size != 3:
            raise ValueError("Vector requires exactly 3 values")
        self._arr = arr

    def __iter__(self):
        return iter(self._arr.tolist())

    def __len__(self):
        return 3

    def __getitem__(self, item):
        return self._arr[item]

    def __array__(self, dtype=None):
        return np.asarray(self._arr, dtype=dtype)

    def copy(self):
        return Vector(self._arr.copy())


class Quaternion:
    def __init__(self, values):
        arr = _as_numpy(values).reshape(-1)
        if arr.size != 4:
            raise ValueError("Quaternion requires exactly 4 values")
        self._arr = arr

    def normalized(self):
        norm = np.linalg.norm(self._arr)
        if norm == 0:
            return Quaternion([1.0, 0.0, 0.0, 0.0])
        return Quaternion(self._arr / norm)

    def to_matrix(self):
        # Blender uses (w, x, y, z), scipy uses (x, y, z, w)
        q = self.normalized()._arr
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return Matrix(rot.as_matrix())

    def __iter__(self):
        return iter(self._arr.tolist())

    def __len__(self):
        return 4

    def __getitem__(self, item):
        return self._arr[item]

    def __array__(self, dtype=None):
        return np.asarray(self._arr, dtype=dtype)


class Euler:
    def __init__(self, values, order='XYZ'):
        arr = _as_numpy(values).reshape(-1)
        if arr.size != 3:
            raise ValueError("Euler requires exactly 3 values")
        self._arr = arr
        self._order = order

    def to_matrix(self):
        rot = Rotation.from_euler(self._order.lower(), self._arr)
        return Matrix(rot.as_matrix())


class Matrix:
    def __init__(self, values):
        arr = _as_numpy(values)
        if arr.shape not in ((3, 3), (4, 4)):
            raise ValueError("Matrix must be 3x3 or 4x4")
        self._arr = arr.copy()

    @staticmethod
    def Translation(vec):
        t = _as_numpy(vec).reshape(-1)
        if t.size != 3:
            raise ValueError("Translation vector requires 3 values")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = t
        return Matrix(mat)

    def resize_4x4(self):
        if self._arr.shape == (4, 4):
            return
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self._arr
        self._arr = out

    def invert_safe(self):
        self._arr = np.linalg.inv(self._arr)

    def decompose(self):
        if self._arr.shape != (4, 4):
            raise ValueError("decompose expects a 4x4 matrix")
        t = Vector(self._arr[:3, 3])
        rot = Rotation.from_matrix(self._arr[:3, :3]).as_quat()
        q = Quaternion([rot[3], rot[0], rot[1], rot[2]])
        scale = Vector([1.0, 1.0, 1.0])
        return t, q, scale

    def copy(self):
        return Matrix(self._arr.copy())

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Matrix multiplication requires Matrix operands")
        return Matrix(np.matmul(self._arr, other._arr))

    def __iter__(self):
        return iter(self._arr.tolist())

    def __len__(self):
        return self._arr.shape[0]

    def __getitem__(self, item):
        return self._arr[item]

    def __array__(self, dtype=None):
        return np.asarray(self._arr, dtype=dtype)
