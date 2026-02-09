import pytest
import torch
from torchquad import Trapezoid
from pathlib import Path
from deadleaves.distributions import PowerLaw, Cosine, ExpCosine, Constant, Image

torch.manual_seed(1)  # set seed for reproducibility

# --- Test: Powerlaw distribution -----------------------------

test_parameters_PowerLaw = [
    (1.0, 10.0, 3.0),  # basic
    (1.0, 1000.0, 2.5),  # wide
    (1.0, 5.5, 3.0),  # narrow
    (1e-3, 1.0, 2.0),  # small
    (100.0, 1000.0, 3.0),  # large
    (1.0, 10.0, 2.3),  # non-integer k
    (1.0, 10.0, 1.1),  # near invalid k
]


@pytest.mark.parametrize("low, high, k", test_parameters_PowerLaw)
def test_PowerLaw(low, high, k):
    dist = PowerLaw(low, high, k)
    x_values = torch.linspace(0.8 * low, 1.2 * high, steps=1000)
    # test density function
    # non-negativity
    assert (dist.pdf(x_values) >= 0).all()
    # normalization
    assert Trapezoid().integrate(
        dist.pdf, 1, 10000, integration_domain=[[low, high]]
    ) == pytest.approx(1, abs=0.05)
    # test cumulative density function
    # boundary conditions
    assert dist.cdf(torch.tensor(low)) == pytest.approx(0, abs=1.0e-6)
    assert dist.cdf(torch.tensor(high)) == pytest.approx(1, abs=1.0e-5)
    # monotonicity
    assert torch.all(dist.cdf(x_values)[1:] >= dist.cdf(x_values)[:-1])
    # test inverse cumulative density function
    p_values = torch.linspace(0.01, 0.99, 100)
    x = dist.icdf(p_values)
    reconstructed_p_values = dist.cdf(x)
    assert torch.allclose(reconstructed_p_values, p_values, atol=1e-5)
    x_values = torch.linspace(low, high, steps=100)
    p = dist.cdf(x_values)
    reconstructed_x_values = dist.icdf(p)
    assert torch.allclose(reconstructed_x_values, x_values, rtol=1e-3)
    # test sampling function
    samples = dist.sample(1000000)
    hist, bins = torch.histogram(samples, bins=100, range=(low, high), density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    pdf_vals = dist.pdf(bin_centers)
    assert torch.allclose(hist[5:-5], pdf_vals[5:-5], atol=1e-2)


test_invalid_parameters_PowerLaw = [
    (-1, 2, 3),  # negative lower bound
    (2, 1, 3),  # upper bound smaller than lower bound
    (1, 2, -1),  # negative exponent
]


@pytest.mark.parametrize("low, high, k", test_invalid_parameters_PowerLaw)
def test_PowerLaw_invalid_args(low, high, k):
    with pytest.raises(ValueError):
        PowerLaw(low, high, k)


# --- Test: Cosine distribution -----------------------------------------------

test_parameters_Cosine = [
    (0.0, 1),  # amplitude at lower bound, minimal frequency
    (0.0, 4),  # amplitude at bound, default frequency
    (0.3, 1),  # interior amplitude, minimal frequency
    (0.4, 10),  # interior amplitude, large frequency
    (0.5, 4),  # default values
    (1, 4),  # amplitude at upper bound, default frequency
    (1, 10),  # amplitude at upper bound, large frequency
]


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
    x = dist.ppf(p_values)
    reconstructed_p_values = dist.cdf(x)
    assert torch.allclose(reconstructed_p_values, p_values, atol=1e-5)
    x_values = torch.linspace(-torch.pi, torch.pi, 100)
    p = dist.cdf(x_values)
    reconstructed_x_values = dist.ppf(p)
    assert torch.allclose(reconstructed_x_values, x_values, atol=1e-3)
    # test sampling function
    samples = dist.sample(1000000)
    hist, bins = torch.histogram(
        samples, bins=100, range=(-torch.pi, torch.pi), density=True
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    pdf_vals = dist.pdf(bin_centers)
    assert torch.allclose(hist, pdf_vals, atol=1e-2)


test_invalid_parameters_Cosine = [
    (-0.1, 4),  # amplitude below lower bound
    (2.0, 4),  # amplitude above upper bound
    (0.5, 0),  # frequency zero (not positive)
    (0.5, -1),  # frequency negative
    (0.5, 2.5),  # frequency not an integer
]


@pytest.mark.parametrize("amplitude, frequency", test_invalid_parameters_Cosine)
def test_Cosine_invalid_args(amplitude, frequency):
    with pytest.raises(ValueError):
        Cosine(amplitude, frequency)


# --- Test: Exponential cosine distribution -----------------------------------


test_parameters_ExpCosine = [
    (
        1,
        1.0,
    ),  # minimal parameters
    (4, 3.0),  # default values
    (2, 0.5),  # small positive values
    (5, 3.0),  # larger valid values
    (1, 1e-6),  # very small positive values, boundary test
]


@pytest.mark.parametrize("frequency, exponential_constant", test_parameters_ExpCosine)
def test_ExpCosine(frequency, exponential_constant):
    dist = ExpCosine(frequency, exponential_constant)
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
    assert dist.cdf(torch.tensor(-torch.pi)) == pytest.approx(0, abs=1.0e-3)
    assert dist.cdf(torch.tensor(torch.pi)) == pytest.approx(1, abs=1.0e-3)
    # monotonicity
    assert torch.all(dist.cdf(x_values)[1:] >= dist.cdf(x_values)[:-1])
    # test inverse cumulative density function
    p_values = torch.linspace(0.01, 0.99, 100)
    x = dist.ppf(p_values)
    reconstructed_p_values = dist.cdf(x)
    assert torch.allclose(reconstructed_p_values, p_values, atol=1e-3)
    x_values = torch.linspace(-torch.pi, torch.pi, 100)
    p = dist.cdf(x_values)
    reconstructed_x_values = dist.ppf(p)
    assert torch.allclose(reconstructed_x_values[:-1], x_values[:-1], atol=1e-3)
    # test sampling function
    samples = dist.sample(1000000)
    hist, bins = torch.histogram(
        samples, bins=100, range=(-torch.pi, torch.pi), density=True
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    pdf_vals = dist.pdf(bin_centers)
    assert torch.allclose(hist, pdf_vals, rtol=0.1)


test_invalid_parameters_ExpCosine = [
    (0, 3.0),  # frequency zero
    (-2, 3.0),  # frequency negative
    (2.5, 3.0),  # frequency not integer
    (4, 0.0),  # exponential_constant zero
    (4, -1.0),  # exponential_constant negative
]


@pytest.mark.parametrize(
    "frequency, exponential_constant", test_invalid_parameters_ExpCosine
)
def test_ExpCosine_invalid_args(frequency, exponential_constant):
    with pytest.raises(ValueError):
        ExpCosine(frequency, exponential_constant)


# --- Test: Constant distribution ---------------------------------------------


test_parameters_Constant = [
    0.0,  # zero
    1.0,  # positive integer as float
    -1.0,  # negative value
    3.1415,  # positive float
    -2.718,  # negative float
    1e-4,  # small positive
    -1e-4,  # small negative
    1e4,  # large positive
    -1e4,  # large negative
]


@pytest.mark.parametrize("value", test_parameters_Constant)
def test_Constant(value):
    dist = Constant(value)
    x_values = torch.linspace(value - 1, value + 1, steps=1001)
    x_values[500] = value
    # test density function
    # non-negativity
    assert (dist.pdf(x_values) >= 0).all()
    # pdf is zero everywhere except at value
    assert (
        dist.pdf(x_values).sum() == 1.0
        or dist.pdf(x_values)[value == x_values].item() == 1.0
    )
    # test cumulative density function
    # boundary conditions
    assert dist.cdf(torch.tensor(value - 0.1)) == pytest.approx(0, abs=1e-6)
    assert dist.cdf(torch.tensor(value)) == pytest.approx(1, abs=1e-6)
    # monotonicity
    assert torch.all(dist.cdf(x_values)[1:] >= dist.cdf(x_values)[:-1])
    # test inverse cumulative density function
    p_values = torch.linspace(0, 1, 100)
    x_from_icdf = dist.icdf(p_values)
    assert torch.all(x_from_icdf == value)
    # test sampling function
    samples = dist.sample(100)
    assert torch.all(samples == value)


# --- Test: Image distribution ------------------------------------------------


def test_Image(tmp_path):
    # create a temporary folder with image files
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "a.png").write_bytes(b"123")
    (img_dir / "b.jpg").write_bytes(b"456")
    (img_dir / "c.jpeg").write_bytes(b"789")
    dist = Image(img_dir)
    # test that all files are discovered
    assert len(dist.files) == 3
    # test sampling function
    samples = dist.sample(5)
    assert len(samples) == 5
    for sample in samples:
        assert Path(sample).suffix.lower() in [".png", ".jpg", ".jpeg"]
        assert Path(sample).exists()
    # test uniform distribution of images in sample
    samples = dist.sample(1000)
    counts = {f: samples.count(f) for f in dist.files}
    avg = sum(counts.values()) / len(dist.files)
    for c in counts.values():
        assert abs(c - avg) < 3 * (avg**0.5)  # loose chi-square-ish check


def test_Image_invalid_args(tmp_path):
    with pytest.raises(TypeError):
        Image(None)  # Invalid input type
    with pytest.raises(FileNotFoundError):
        Image("./nonexistent")  # Non existent directory
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        Image(empty)  # No images in directory
