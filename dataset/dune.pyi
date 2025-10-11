from typing import Tuple
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

class DuneSediment:
    def __init__(
        self,
        nx: int,
        ny: int,
        r_min: float,
        r_max: float,
        wind: Tuple[float, float],
        cell_size: Tuple[float, float] = (1, 1),
        vegetation_on: bool = False,
        abrasion_on: bool = False,
    ) -> None: ...

    nx: int
    ny: int
    vegetation_on: bool
    abrasion_on: bool

    bedrock: FloatArray
    sediments: FloatArray
    vegetation: FloatArray
    wind_x: FloatArray
    wind_y: FloatArray
    bedrock_hardness: FloatArray

    def step(self) -> None: ...
