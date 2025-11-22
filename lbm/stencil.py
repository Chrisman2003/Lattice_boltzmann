import numpy as np


class Stencil:
    def __init__(self, d: int, q: int):
        self._d = d
        self._q = q
        self._w = None
        self._c = None

        if self.d == 2 and self.q == 9:
            self.__d2q9()
        elif self.d == 3 and self.q == 19:
            self.__d3q19()
        else:
            raise Exception(f'Invalid choice of DdQq: "D{self.d}Q{self.q}"')

    @property
    def d(self):
        return self._d

    @property
    def q(self):
        return self._q

    @property
    def w(self):
        return self._w

    @property
    def c(self):
        return self._c

    def __d2q9(self):
        self._w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
        self._c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)

    def __d3q19(self):
        self._w = np.array([1/3,
                            1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                            1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
                            1/36, 1/36, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
        self._c = np.array([[0, 0, 0],
                            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                            [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]], dtype=np.int32)


