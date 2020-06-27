import numpy as np
import pytest

from growth import grow


@pytest.fixture
def test_settings():
    return grow.COOL_SETTINGS["snake"]


@pytest.mark.parametrize("n_return", [1, 2, 3, 4, 5])
def test_run_reaction_diffusion(test_settings, n_return):
    """Make sure the function actually returns the right number of snapshots"""

    results = grow.run_reaction_diffusion(n=10, n_to_return=n_return, **test_settings)

    assert len(results) == n_return


@pytest.mark.parametrize("grid_size", [(100, 100), (100, 200), (200, 100)])
def test_run_reaction_diffusion_grid_size(test_settings, grid_size):

    results = grow.run_reaction_diffusion(
        n=10, grid_size=grid_size, n_to_return=1, **test_settings
    )
    assert results[0][0].shape == grid_size


def test_run_reaction_diffusion_raises_grid_size(test_settings):

    mask = np.zeros((10, 11))
    with pytest.raises(AssertionError, match="The mask size and grid size don't match"):
        grow.run_reaction_diffusion(
            n=10, grid_size=(10, 10), mask=mask, n_to_return=1, **test_settings
        )


def test_run_reaction_diffusion_raises_param_length(test_settings):

    test_settings.update(dict(dA=np.random.random(11)))

    with pytest.raises(
        AssertionError, match="the wrong dimensions. Should be of length 10"
    ):
        grow.run_reaction_diffusion(
            n=10, grid_size=(10, 10), n_to_return=1, **test_settings
        )
