import numpy as np
from typing import Tuple

class DuneSediment:
    nx: int
    ny: int
    vegetation_on: bool
    abrasion_on: bool

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
    @property
    def bedrock(self) -> np.ndarray: ...
    @property
    def sediments(self) -> np.ndarray: ...
    @property
    def vegetation(self) -> np.ndarray: ...
    def step(self) -> None: ...
