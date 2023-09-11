from __future__ import annotations

from gpytorch.distributions import Distribution, MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel, Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ExactGP
from torch import Tensor


class SimpleGpUnemploymentModel(ExactGP):  # type: ignore
    def __init__(self, likelihood: GaussianLikelihood) -> None:
        super().__init__(None, None, likelihood)
        self._mean_module = ConstantMean()
        self._cov_module = ScaleKernel(MaternKernel(nu=0.5))
        self._cov_module.base_kernel.lengthscale = 25

    def forward(self, x: Tensor) -> Distribution:
        mean_x = self._mean_module(x)
        cov_x = self._cov_module(x)
        return MultivariateNormal(mean_x, cov_x)

    @property
    def mean_module(self) -> Mean:
        return self._mean_module

    @property
    def covariance_module(self) -> Kernel:
        return self._cov_module
