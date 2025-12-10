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

    Raises:
            NotImplementedError:
                Class contains empty methods for
                    - initialization (__init__)
                    - probability density function (pdf)
                    - inverse cumulative function (icdf)
    """

    _batch_shape: torch.Size = torch.Size()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def _validate_args(self):
        """Validate input arguments.

        Raises:
            ValueError
        """
        for param, constraint in self.arg_constraints.items():
            value = getattr(self, param)
            valid = constraint.check(value)
            if not torch._is_all_true(valid):
                raise ValueError(
                    f"Expected parameter {param} "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to satisfy the constraint {repr(constraint)}, "
                    f"but found invalid values:\n{value}"
                )

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the probability density function evaluated at *x*.

        Args:
            x (torch.Tensor):
                Value(s) to evaluate.
        """
        raise NotImplementedError

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (approximation of) the cumulative density function evaluated at *x*.
        If not implemented this method will compute the cdf as approximation
        via the pdf.

        Args:
            x (torch.Tensor):
                Value(s) to evaluate.
        """
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
        """Returns the inverse cumulative density function evaluated at *p*.

        Args:
            p (torch.Tensor):
                Probability value(s) to evaluate.
        """
        raise NotImplementedError

    def ppf(self, p: torch.Tensor) -> torch.Tensor:
        """Returns the percent point function, i.e. approximation of inverse cumulative
        function evaluated at *p*.

        Args:
            p (torch.Tensor): Probability value(s) to evaluate.
        """
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
        """Generates a sample from the distribution.

        Args:
            n (int, optional): Number of samples. Defaults to 1.
        """
        try:
            return self.icdf(torch.rand(n))
        except NotImplementedError:
            return self.ppf(torch.rand(n))


class PowerLaw(BaseDistribution):
    """Distribution with density that follows the power law.

    Args:
        low (float):
            Minimal allowed value.
        high (float):
            Maximal allowed value.
        k (float, optional):
            Power law exponent. Defaults to 3.
    """

    @property
    def arg_constraints(self) -> dict:
        return {
            "k": constraints.positive,
            "low": constraints.positive,
            "high": constraints.greater_than(self.low),
        }

    def __init__(self, low: float, high: float, k: float = 3) -> None:
        self.low, self.high, self.k = broadcast_all(low, high, k)
        self.scale_factor: torch.Tensor = self.low ** (1 - self.k) - self.high ** (
            1 - self.k
        )
        self._validate_args()

    @property
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
    """Distribution with density that follows the cosine

    Args:
        amplitude (float):
            Amplitude of cosine. Value must be between 0.0 and 1.0.
            Defaults to 0.5.
        frequency (int):
            Frequency of cosine. Defaults to 4,
            i.e. peaks at the cardinals.
    """

    @property
    def support(self):
        return constraints.interval(-torch.pi, torch.pi)

    @property
    def arg_constraints(self):
        return {
            "amplitude": constraints.interval(0, 1),
            "frequency": constraints.positive_integer,
        }

    def __init__(self, amplitude: float = 0.5, frequency: int = 4) -> None:
        self.amplitude, self.frequency = broadcast_all(amplitude, frequency)
        self._validate_args()

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
    """Distribution with cosine density and exponential peaks.

    Args:
        amplitude (float):
            Amplitude of density function. Value must be positive.
            Defaults to 1.
        frequency (int):
            Frequency of cosine. Defaults to 4,
            i.e. peaks at the cardinals.
        exponential_constant (float):
            Growth constant of exponential component.
            Larger values generate stronger peaks. Negative values invert the peaks.
            Defaults to 3.
    """

    @property
    def support(self):
        return constraints.interval(-torch.pi, torch.pi)

    @property
    def arg_constraints(self):
        return {
            "amplitude": constraints.positive,
            "frequency": constraints.positive_integer,
            "exponential_constant": constraints.positive,
        }

    def __init__(
        self,
        amplitude: float = 1,
        frequency: int = 4,
        exponential_constant: float = 3,
    ) -> None:
        self.amplitude, self.frequency, self.exponential_constant = broadcast_all(
            amplitude, frequency, exponential_constant
        )
        self._validate_args()

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        def f(x):
            return self.amplitude * torch.exp(
                -self.exponential_constant
                * torch.sqrt(1 - torch.cos(self.frequency * x))
            )

        integral = Trapezoid().integrate(
            f,
            dim=1,
            integration_domain=[[self.support.lower_bound, self.support.upper_bound]],
        )
        d = torch.where(self.support.check(x), f(x) / integral, 0)
        return d


class Constant(Distribution):
    """Distribution class which return a constant value.

    Args:
        value (float):
            Constant value to be return in each sampling.
    """

    @property
    def arg_constraints(self):
        return {"value": constraints.real}

    @property
    def support(self):
        return constraints.real

    _batch_shape: torch.Size = torch.Size()

    def __init__(self, value: float) -> None:
        self.value: float = value

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the probability density function evaluated at *x*.

        Args:
            x (torch.Tensor):
                Value(s) to evaluate.
        """
        return torch.where(x == self.value, torch.tensor(1.0), torch.tensor(0.0))

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the cumulative density function evaluated at *x*.

        Args:
            x (torch.Tensor):
                Value(s) to evaluate.
        """
        return torch.where(
            torch.tensor(x < self.value), torch.tensor(0.0), torch.tensor(1.0)
        )

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        """Returns the inverse cumulative density function evaluated at *p*.

        Args:
            p (torch.Tensor):
                Probability value(s) to evaluate.
        """
        return self.value * torch.ones_like(p)

    def sample(self, n=1) -> torch.Tensor:
        """Generates a sample from the distribution.

        Args:
            n (int, optional): Number of samples. Defaults to 1.
        """
        return self.value * torch.ones(n)


class Image(Distribution):
    """Distribution to sample images uniformly from an image data set.

    Args:
        dir (Path | str):
            Path to image data set directory.
    """

    @property
    def arg_constraints(self):
        return {"dir": constraints.dependent}

    @property
    def support(self):
        return constraints.real

    _batch_shape: torch.Size = torch.Size()

    def __init__(self, dir: Path | str) -> None:
        if not isinstance(dir, (str, Path)):
            raise TypeError("dir must be a string or Path object.")
        if not Path(dir).exists():
            raise FileNotFoundError(f"Directory {dir} does not exist.")
        self.dir: Path | str = dir
        file_list = []
        for root, _, files in os.walk(self.dir):
            file_list += [os.path.join(root, f) for f in files]
        self.files: list[str] = [
            file for file in file_list if re.search(r"\.(png|jpg|gif|tiff|jpeg)$", file)
        ]

        if len(self.files) == 0:
            raise ValueError(f"No image files found in directory {self.dir}")

        self.n_files: int = len(self.files)

    def sample(self, n=1) -> list[Path]:
        """Draws a sample from the available images.

        Args:
            n (int, optional): Number of samples. Defaults to 1.
        """
        idx = torch.multinomial(
            torch.ones(self.n_files), num_samples=n[0], replacement=True
        )
        return [self.files[i] for i in idx]
