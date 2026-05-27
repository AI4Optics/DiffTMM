import pytest
import torch


@pytest.fixture
def device():
    """CPU device — tests should not require CUDA."""
    return torch.device("cpu")


@pytest.fixture
def wvln_vis(device):
    """A visible-light wavelength vector in micrometers."""
    return torch.tensor([0.45, 0.55, 0.65], device=device)
