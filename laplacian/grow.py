import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap


def discrete_laplacian(X: np.array) -> np.array:
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

    Y = -4 * X

    # by using np.roll, the borders of the matrix 'wrap around'
    Y += np.roll(X, (1, 0), (0, 1))
    Y += np.roll(X, (-1, 0), (0, 1))
    Y += np.roll(X, (0, -1), (0, 1))
    Y += np.roll(X, (0, 1), (0, 1))

    # # a different way 
    # L = -4 * M
    # L = L + np.roll(M, -1)
    # L = L + np.roll(M, 1)
    # L = L + np.roll(M, -1).T
    # L = L + np.roll(M, 1).T

    return Y


def discrete_laplacian_isotropic(M: np.array):
    """Get the discrete Laplacian of matrix M"""
    L = -3 * M
    L += 0.5 * np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += 0.5 * np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += 0.5 * np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += 0.5 * np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    L += 0.25 * np.roll(M, (1, 1), (0, 1))
    L += 0.25 * np.roll(M, (-1, -1), (0, 1))
    L += 0.25 * np.roll(M, (1, -1), (0, 1))
    L += 0.25 * np.roll(M, (-1, 1), (0, 1))

    return L



def reaction_diffusion(
    A: np.array,
    B: np.array,
    dA: float,
    dB: float,
    kill: float, # make this accept a distribution implenting .sample or a float
    feed: float, # ditto here
    delta_t: int =1,
    kind: str="normal",
    mask: Optional[np.array]
) -> Tuple[np.array, np.array]:
    """[summary]

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
    
    if kind == "normal":
        LA = discrete_laplacian(A)
        LB = discrete_laplacian(B)

    elif kind == "isonormal":
        LA = discrete_laplacian_isotropic(A)
        LB = discrete_laplacian_isotropic(B)
    else:
        # raise an error
        pass

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


def setup_grid(
    n_dim: Tuple[int, int] = (100,100), random_influence=0.2, seed: int = 11
) -> Tuple[np.array, np.array]:


    A = (1 - random_influence) * np.ones(
        n_dim
    ) + random_influence * np.random.random(n_dim)

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


def plot(X: np.array, cmap:str="Greys") -> Figure:
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(X, cmap=cmap)
    ax.axis("off")

COOL_SETTINGS: Dict[str, Dict[str, Any]] = dict(
    "lines and dots" = dict(dA=0.16, dB=0.08, kill=0.06, feed=0.0299, delta_t=1) ,
"snake" = dict(dA=0.2, dB=0.1, kill=0.065, feed=0.05, delta_t=2) ,
"mitosis" = dict(dA=0.24, dB=0.08, kill=0.07, feed=0.035, delta_t=1),
"fingerprint" = dict(dA=0.25, dB=0.1, kill=0.065, feed=0.055, delta_t=1) ,
"snowflake maze" = dict(dA=0.2, dB=0.1, kill=0.062, feed=0.05, delta_t=1) ,
"coral maze" = dict(dA=0.2, dB=0.06, kill=0.062, feed=0.05, delta_t=1),
"maze"= dict(dA=0.22, dB=0.06, kill=0.062, feed=0.05, delta_t=1) ,
"brain" = dict(dA=0.2, dB=0.1, kill=0.065, feed=0.05, delta_t=2)
)