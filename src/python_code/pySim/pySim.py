from typing import Optional, Mapping, Tuple, Sequence, NoReturn, Union

import torch
import torch.nn as nn
from torch import Tensor

import diffcloth_py as diffcloth

from .functional import SimFunction, SimFunction_cube2cloth, SimFunction_cloth2cube


class pySim(nn.Module):

    def __init__(self, cppSim: diffcloth.Simulation, optimizeHelper: diffcloth.OptimizeHelper, useFixedPoint: bool) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper
        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(self, x: Tensor, v: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        return SimFunction.apply(x, v, a, self.cppSim, self.optimizeHelper)


class pySim_cloth2cube(nn.Module):

    def __init__(self, cppSim: diffcloth.Simulation, optimizeHelper: diffcloth.OptimizeHelper, useFixedPoint: bool) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper
        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(self, x: Tensor, v: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        return SimFunction_cloth2cube.apply(x, v, a, self.cppSim, self.optimizeHelper)


class pySim_cube2cloth(nn.Module):

    def __init__(self, cppSim: diffcloth.Simulation, optimizeHelper: diffcloth.OptimizeHelper, useFixedPoint: bool) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper
        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(self, x: Tensor, v: Tensor, a: Tensor, p_control: Tensor) -> Tuple[Tensor, Tensor]:
        return SimFunction_cube2cloth.apply(x, v, a, p_control, self.cppSim, self.optimizeHelper)
    