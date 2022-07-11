import numpy as np
from typing import Tuple, Union, Sequence


class ClosedError(PermissionError):
    pass


class GrowableNumpyArray:
    DEF_INIT_CAPACITY = 10
    GROWTH_RATE = 1.5

    def __init__(
        self,
        element_shape: Union[Tuple[int], int],
        dtype: Union[np.dtype, np.generic] = np.float32,
        init_capacity: int = DEF_INIT_CAPACITY
    ):
        if(isinstance(element_shape, int)):
            element_shape = (element_shape,)

        self._capacity = int(init_capacity)
        self._element_shape = element_shape
        self._array = np.zeros((self._capacity, *element_shape), dtype, order='C')
        self._finalized = False
        self._size = 0

    @property
    def view(self) -> np.ndarray:
        return self._array[:self._size]

    def __len__(self) -> int:
        return self._size

    def add(self, elem: Union[Sequence, np.ndarray]):
        if(self._finalized):
            raise ClosedError("Not allowed to grow a finalized array!")

        if(self._size == self._capacity):
            self._capacity = int(np.ceil(self._capacity * self.GROWTH_RATE))
            self._array.resize((self._capacity, *self._element_shape))

        self._array[self._size, :] = elem
        self._size += 1

    def finalize(self) -> np.ndarray:
        self._finalized = True
        return self._array[:self._size]