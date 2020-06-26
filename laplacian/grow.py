#!/usr/bin/env python

import numpy as np
from typing import Tuple, Optional, Dict, Any

# ParameterType = Union[float, Iterable[float]]


def discrete_laplacian(X: np.array, kind: str = "original") -> np.array:
    """Compute the 2-D discrete Laplacian.

    Parameters
    ----------
    X : np.array
        [description]

    Returns
    -------
    np.array
        [description]

    Notes
    -------
    See https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    if kind == "original":

        L = -4 * X

        # by using np.roll, the borders of the matrix 'wrap around'
        L += np.roll(X, (1, 0), (0, 1))
        L += np.roll(X, (-1, 0), (0, 1))
        L += np.roll(X, (0, -1), (0, 1))
        L += np.roll(X, (0, 1), (0, 1))

        # # a different way
        # L = -4 * M
        # L = L + np.roll(M, -1)
        # L = L + np.roll(M, 1)
        # L = L + np.roll(M, -1).T
        # L = L + np.roll(M, 1).T

    elif kind == "isotropic":
        L = -3 * X
        L += 0.5 * np.roll(X, (0, -1), (0, 1))  # right neighbor
        L += 0.5 * np.roll(X, (0, +1), (0, 1))  # left neighbor
        L += 0.5 * np.roll(X, (-1, 0), (0, 1))  # top neighbor
        L += 0.5 * np.roll(X, (+1, 0), (0, 1))  # bottoX neighbor

        L += 0.25 * np.roll(X, (1, 1), (0, 1))
        L += 0.25 * np.roll(X, (-1, -1), (0, 1))
        L += 0.25 * np.roll(X, (1, -1), (0, 1))
        L += 0.25 * np.roll(X, (-1, 1), (0, 1))

    else:
        raise NotImplementedError(
            f"""Type {kind} is not implemented.
                Appropriate kinds are 'normal' and 'isotropic'
                """
        )

    return L


def reaction_diffusion(
    A: np.array,
    B: np.array,
    dA: float,
    dB: float,
    kill: float,
    feed: float,
    delta_t: int = 1,
    kind: str = "normal",
    mask: Optional[np.array] = None,
) -> Tuple[np.array, np.array]:
    """Run one iteration of reaction diffusion.

    Parameters
    ----------
    A : np.array
        Matrix containing proportion of As
    B : np.array
        Matrix containing proportion of Bs
    dA : float
        Diffusion rate of A
    dB : float
        Diffusion rate of B
    kill : float
        Kill rate of A
    feed : float
        Feed rate for B
    delta_t : int
        Size of the time step
    mask : Optional[np.array]
        A mask to apply to the both A and B marices at the end of the growth step.
    kind : str, optional
        Determines the type of Laplacian to use, by default "normal"

    Returns
    -------
    Tuple[np.array, np.array]
        A tuple of the A and B matrices updated

    Raises
    ------
    an
        [description]
    """

    LA = discrete_laplacian(A, kind)
    LB = discrete_laplacian(B)

    # create the feed matrix
    # N2 = A.shape[0] // 2
    # feed_mat = np.full(A.shape, feed)
    # x = np.random.uniform(feed - 0.02, feed + 0.02, A.shape[0])
    # x.sort()
    # feed_mat[:,] = x

    # # create the kill matrix
    # N2 = A.shape[0] // 2
    # kill_mat = np.full(A.shape, kill)
    # x = np.random.uniform(kill-0.007, kill+0.01, A.shape[0])
    # x.sort()
    # kill_mat[:,] = x

    # Now apply the update formula
    diff_A = (dA * LA - A * B ** 2 + feed * (1 - A)) * delta_t
    diff_B = (dB * LB + A * B ** 2 - (kill + feed) * B) * delta_t

    A += diff_A
    B += diff_B

    if mask is not None:
        A[mask] = 0
        B[mask] = 0

    return A, B


def run_reaction_diffusion(
    dA: ParameterType,
    dB: ParameterType,
    kill: ParameterType,
    feed: ParameterType,
    grid_size: Tuple[int, int] = (100, 100),
    kind: str = "normal",
    mask: np.array = None,
):
    """Coordinate a series of reaction diffusion steps."""
    pass


def setup_grid(
    n_dim: Tuple[int, int] = (100, 100), random_influence=0.2, seed: int = 11
) -> Tuple[np.array, np.array]:

    A = (1 - random_influence) * np.ones(n_dim) + random_influence * np.random.random(
        n_dim
    )

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random(n_dim)

    #     A = np.random.normal(0.7, 0.05, (n_cells, n_cells))
    #     B = np.random.normal(0.2, 0.05, (n_cells, n_cells))

    # Now let's add a disturbance in the center
    N2 = n_dim[0] // 2
    r = n_dim[0] // 20

    A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
    B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

    # pick some cells at random
    #     random_points_x = np.random.choice(n_cells,replace=False, size=n_cells)
    #     random_points_y = np.random.choice(n_cells,replace=False, size=n_cells)
    #     A[random_points_x, random_points_y] = 0.5
    #     B[random_points_x, random_points_y] = 0.5

    # create circle instead
    #     y,x = np.ogrid[-N2:n_cells-N2, -N2:n_cells-N2]
    #     mask = x*x + y*y <= r*r
    #     A[mask] = 0.5
    #     B[mask] = 0.2

    return A, B


def simulate():
    pass


COOL_SETTINGS = {
    "lines and dots": dict(dA=0.16, dB=0.08, kill=0.06, feed=0.0299, delta_t=1),
    "snake": dict(dA=0.2, dB=0.1, kill=0.065, feed=0.05, delta_t=2),
    "mitosis": dict(dA=0.24, dB=0.08, kill=0.07, feed=0.035, delta_t=1),
    "fingerprint": dict(dA=0.25, dB=0.1, kill=0.065, feed=0.055, delta_t=1),
    "snowflake maze": dict(dA=0.2, dB=0.1, kill=0.062, feed=0.05, delta_t=1),
    "coral maze": dict(dA=0.2, dB=0.06, kill=0.062, feed=0.05, delta_t=1),
    "maze": dict(dA=0.22, dB=0.06, kill=0.062, feed=0.05, delta_t=1),
    "brain": dict(dA=0.2, dB=0.1, kill=0.065, feed=0.05, delta_t=2),
}
