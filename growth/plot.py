import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

mustard_colors = [(241 / 255, 240 / 255, 226 / 255), (216 / 255, 174 / 255, 72 / 255)]
colors_pom = [(241 / 255, 240 / 255, 226 / 255), (108 / 255, 40 / 255, 49 / 255,)]

bean_cols = ["#D59224", "#D08234", "#BC5F43", "#965A34"]

fire_cols = [
    "#962C19",
    "#BC2414",
    "#DB421F",
    "#E26B1E",
    "#E98F15",
    "#F2B37F",
    "#F9E5C8",
]

buda_cols = [
    "#F1BB7B",
    "#FD6467",
    "#5B1A18",
]

straw_cols = ["#962C19", "#ffffff"]

green_cols = ["#357686", "#F1F0E2"]

mustard = LinearSegmentedColormap.from_list("mustard", mustard_colors, N=100)
pom = LinearSegmentedColormap.from_list("pom", colors_pom, N=100)
bean = LinearSegmentedColormap.from_list("pom", bean_cols, N=100)
fire = LinearSegmentedColormap.from_list("fire", fire_cols, N=100)
buda = LinearSegmentedColormap.from_list("buda", buda_cols, N=100)
straw = LinearSegmentedColormap.from_list("straw", straw_cols, N=100)
green = LinearSegmentedColormap.from_list("green", green_cols, N=100)


def plot(X: np.array, cmap: str = "Greys") -> Figure:
    fig = Figure()
    ax = fig.subplots()
    ax.imshow(X, cmap=cmap)
    ax.axis("off")


def write_matrix(X: np.array, path: str = None) -> None:

    assert path is not None, "name required"

    M = X.copy()

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)

    # Shaded rendering
    light = colors.LightSource(azdeg=315, altdeg=10)
    M = light.shade(
        M, cmap=pom, vert_exag=1.5, norm=colors.PowerNorm(0.3), blend_mode="hsv"
    )
    ax.imshow(M, interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path)
