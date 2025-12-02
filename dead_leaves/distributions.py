import torch
import os
import re
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torchquad import Trapezoid
from pathlib import Path


class BaseDistribution(Distribution):
    """Base class for custom distributions

    Methods:
        - pdf: Probability density function.
        - cdf: (Approximation of) cumulative density function.
        - icdf: Inverse cumulative density function.
        - ppf: Percent point function, i.e. approximation of
            inverse cumulative function.
        - sample: Draw sample(s) from the distribution.

    Raises:
            NotImplementedError: Class contains empty methods for
                - initialization (__init__)
                - probability density function (pdf)
                - inverse cumulative function (icdf)
    """

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x_grid = torch.linspace(
            self.support.lower_bound, self.support.upper_bound, steps=10000
        )
        pdf_values = self.pdf(x_grid)
        idx = torch.searchsorted(x_grid, x, right=True)
        idx = torch.clamp(idx, 1, len(x_grid) - 1)

        cdf_values = torch.cumsum(pdf_values, dim=0)
        cdf_values = cdf_values - cdf_values[0]
        cdf_values = cdf_values / cdf_values[-1]
        bin_means = 0.5 * (cdf_values[idx - 1] + cdf_values[idx])
        p = torch.where(x <= self.support.lower_bound, 0, 1)
        p = torch.where(
            self.support.check(x),
            bin_means,
            p,
        )

        return p

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ppf(self, p: torch.Tensor) -> torch.Tensor:
        x_grid = torch.linspace(
            self.support.lower_bound, self.support.upper_bound, steps=10000
        )
        cdf_values = self.cdf(x_grid)

        idx = torch.searchsorted(cdf_values, p, right=True)
        idx = torch.clamp(idx, 1, len(x_grid) - 1)

        return x_grid[idx - 1] + (p - cdf_values[idx - 1]) * (
            x_grid[idx] - x_grid[idx - 1]
        ) / (cdf_values[idx] - cdf_values[idx - 1])

    def sample(self, n=1) -> torch.Tensor:
        try:
            return self.icdf(torch.rand(n))
        except NotImplementedError:
            return self.ppf(torch.rand(n))


class PowerLaw(BaseDistribution):
    """Distribution with density that follows the power law

    Args:
        - low (float): Minimal allowed value.
        - high (float): Maximal allowed value.
        - k (float, optional): Power law exponent. Defaults to 3.

    Raises:
        ValueError: Boundaries must construct a non-empty positive domain.
    """

    @property
    def arg_constraints(self):
        return {
            "k": constraints.positive,
            "low": constraints.positive,
            "high": constraints.greater_than(self.low),
        }

    def __init__(self, low: float, high: float, k: float = 3) -> None:
        if low >= high:
            raise ValueError("Minimal x value must be smaller than maximal x value.")
        if low <= 0:
            raise ValueError("Minimal x value must be greater 0.")
        self.low, self.high, self.k = broadcast_all(low, high, k)
        self.scale_factor = self.low ** (1 - self.k) - self.high ** (1 - self.k)
        super().__init__(validate_args=True)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.where(
            self.support.check(x),
            (self.k - 1) / (self.scale_factor * (x**self.k)),
            0,
        )
        return d

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        p = torch.where(x <= self.low, 0, 1)
        p = torch.where(
            self.support.check(x),
            (self.low ** (1 - self.k) - x ** (1 - self.k)) / self.scale_factor,
            p,
        )
        return p

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        x = (self.low ** (1 - self.k) - p * self.scale_factor) ** (1 / (1 - self.k))
        return x


class Cosine(BaseDistribution):
    """Distribution with density that follows that cosine

    Args:
        - amplitude (float): Amplitude of cosine. Value must be between 0.0 and 1.0.
            Defaults to 0.5.
        - frequency (float): Frequency of cosine. Defaults to 4,
            i.e. peaks at the cardinals.

    Raises:
            ValueError: Invalid amplitude values.
    """

    @property
    def support(self):
        return constraints.interval(-torch.pi, torch.pi)

    @property
    def arg_constraints(self):
        return {"amplitude": constraints.interval(0, 1)}

    def __init__(self, amplitude: float = 0.5, frequency: float = 4) -> None:
        self.amplitude = amplitude
        self.frequency = frequency

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.where(
            self.support.check(x),
            0.5 * (1 + self.amplitude * torch.cos(self.frequency * x)) / torch.pi,
            0,
        )
        return d

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.where(x <= -torch.pi, 0, 1)
        p = torch.where(
            self.support.check(x),
            0.5
            + 0.5
            * (x + (self.amplitude / self.frequency) * torch.sin(self.frequency * x))
            / torch.pi,
            p,
        )
        return p


class ExpCosine(BaseDistribution):
    """Distribution with cosine density with exponential peaks.

    Args:
        - amplitude (float): Amplitude of density function. Value must be non-negative.
            Defaults to 2.
        - frequency (float): Frequency of cosine. Defaults to 4.
        - exponential_constant (float): Growth constant of exponential component.
            Larger values generate stronger peaks. Defaults to 3.
    """

    def __init__(
        self,
        amplitude: float = 2,
        frequency: float = 4,
        exponential_constant: float = 3,
    ) -> None:
        if amplitude < 0:
            raise ValueError("Amplitude must be non-negative.")
        self.amplitude = amplitude
        self.frequency = frequency
        self.exponential_constant = exponential_constant

    arg_constraints = {}

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        x_range_cond = (x <= torch.pi) & (x >= -torch.pi)

        def f(x):
            return (
                self.amplitude
                * torch.exp(
                    -self.exponential_constant
                    * torch.sqrt(1 - torch.cos(self.frequency * x))
                )
                + 0.5
            )

        integral = Trapezoid().integrate(
            f, dim=1, integration_domain=[[-torch.pi, torch.pi]]
        )
        d = torch.where(x_range_cond, f(x) / integral, 0)
        return d


class Constant(Distribution):
    """Distribution class which return a constant value.

    Args:
        - value: Constant value to be return in each sampling.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n=1) -> torch.Tensor:
        return self.value * torch.ones(n)


class Image(Distribution):
    """Distribution to sample images uniformly from an image data set.

    Args:
        - dir: Path to image data set directory.
    """

    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.files = []
        for root, _, files in os.walk(self.dir):
            self.files += [os.path.join(root, f) for f in files]
        self.files = [
            file
            for file in self.files
            if re.search(r"\.(png|jpg|gif|tiff|jpeg)$", file)
        ]
        self.n_files = len(self.files)

    def sample(self, n=1) -> Path:
        idx = torch.multinomial(
            torch.ones(self.n_files), num_samples=n, replacement=True
        )
        return self.files[idx]
