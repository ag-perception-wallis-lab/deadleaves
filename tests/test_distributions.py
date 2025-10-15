import pytest
import torch
from torchquad import Trapezoid
from dead_leaves.distributions import PowerLaw, Cosine


test_parameters_PowerLaw = [
    (1.0, 10.0, 3.0),  # basic
    (1.0, 1000.0, 2.5),  # wide
    (1.0, 1.01, 3.0),  # narrow
    (1e-3, 1.0, 2.0),  # small
    (100.0, 1000.0, 3.0),  # large
    (1.0, 10.0, 2.3),  # non-integer k
    (1.0, 10.0, 1.1),  # near invalid k
]


@pytest.mark.parametrize("x_min, x_max, k", test_parameters_PowerLaw)
def test_PowerLaw(x_min, x_max, k):
    dist = PowerLaw(x_min, x_max, k)
    x_values = torch.linspace(0.8 * x_min, 1.2 * x_max, steps=1000)
    # test density function
    # non-negativity
    assert (dist.pdf(x_values) >= 0).all()
    # normalization
    assert Trapezoid().integrate(
        dist.pdf, 1, 10000, integration_domain=[[x_min, x_max]]
    ) == pytest.approx(1, abs=0.05)
    # test cumulative density function
    # boundary conditions
    assert dist.cdf(torch.tensor(x_min)) == pytest.approx(0, abs=1.0e-6)
    assert dist.cdf(torch.tensor(x_max)) == pytest.approx(1, abs=1.0e-5)
    # monotonicity
    assert torch.all(dist.cdf(x_values)[1:] >= dist.cdf(x_values)[:-1])
    # test inverse cumulative density function
    p_values = torch.linspace(0.01, 0.99, 100)
    x = dist.icdf(p_values)
    reconstructed_p_values = dist.cdf(x)
    assert torch.allclose(reconstructed_p_values, p_values, atol=1e-5)
    x_values = torch.linspace(x_min, x_max, steps=100)
    p = dist.cdf(x_values)
    reconstructed_x_values = dist.icdf(p)
    assert torch.allclose(reconstructed_x_values, x_values, rtol=1e-3)
    # test sampling function
    if x_min - x_max > 1:
        samples = dist.sample(1000000)
        hist, bins = torch.histogram(
            samples, bins=100, range=(x_min, x_max), density=True
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        pdf_vals = dist.pdf(bin_centers)
        assert torch.allclose(hist[5:-5], pdf_vals[5:-5], atol=1e-2)


def test_PowerLaw_invalid_args():
    pass


test_parameters_Cosine = [(0.5, 4), (1, 4)]


@pytest.mark.parametrize("amplitude, frequency", test_parameters_Cosine)
def test_Cosine(amplitude, frequency):
    dist = Cosine(amplitude, frequency)
    x_values = torch.linspace(-4, 4, steps=1000)
    # test density function
    # non-negativity
    assert (dist.pdf(x_values) >= 0).all()
    # normalization
    assert Trapezoid().integrate(
        dist.pdf, 1, 10000, integration_domain=[[-4, 4]]
    ) == pytest.approx(1, abs=0.05)
    # test cumulative density function
    # boundary conditions
    assert dist.cdf(torch.tensor(-torch.pi)) == pytest.approx(0, abs=1.0e-6)
    assert dist.cdf(torch.tensor(torch.pi)) == pytest.approx(1, abs=1.0e-5)
    # monotonicity
    assert torch.all(dist.cdf(x_values)[1:] >= dist.cdf(x_values)[:-1])
    # test inverse cumulative density function
    p_values = torch.linspace(0.01, 0.99, 100)
    x = dist.icdf(p_values)
    reconstructed_p_values = dist.cdf(x)
    assert torch.allclose(reconstructed_p_values, p_values, atol=1e-5)
