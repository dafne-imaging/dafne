"""
Generate a matplotlib colormap from a chosen color, spreading N_COLORS
additional hues evenly around the color wheel while keeping them outside
a FORBIDDEN_ANGLE exclusion zone centered on the chosen color's hue.

Colormap layout:
    index 0            -> fully transparent [0, 0, 0, 0]  (e.g. "background/none")
    index 1            -> the chosen color
    index 2 .. N+1     -> the N_COLORS generated colors
"""

import colorsys
from typing import Sequence, Union

from matplotlib.colors import ListedColormap, to_rgb

ColorLike = Union[str, Sequence[float]]

STANDARD_FORBIDDEN_ANGLE = 60

def generate_colormap(
    chosen_color: ColorLike,
    n_colors: int,
    forbidden_angle: float = STANDARD_FORBIDDEN_ANGLE,
    name: str = "hue_compass",
) -> ListedColormap:
    """
    Build a ListedColormap: [transparent, chosen_color, *N_COLORS derived colors].

    Parameters
    ----------
    chosen_color : ColorLike
        Any matplotlib-recognized color spec (hex string, named color,
        or an (r, g, b) / (r, g, b, a) tuple in 0-1 range).
    n_colors : int
        Number of additional colors to generate. Must be > 1.
    forbidden_angle : float
        Half-width, in degrees, of the excluded hue sector centered on the
        chosen color's hue. Must be in [0, 179]. If 0, no sector is
        excluded and the generated colors are spread uniformly around the
        full circumference of the color wheel.
    name : str
        Name assigned to the returned colormap.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap with 2 + n_colors entries as described above.
    """
    if n_colors <= 1:
        n_colors = 2
    if not (0 <= forbidden_angle <= 179):
        raise ValueError(f"forbidden_angle must be in [0, 179], got {forbidden_angle}")

    # Parse the chosen color and pull out its HSV representation.
    r, g, b = to_rgb(chosen_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360.0

    alpha = 1.0

    if not isinstance(chosen_color, str) and len(chosen_color) > 3:
        alpha = chosen_color[3]


    derived = []
    if forbidden_angle == 0:
        # No exclusion zone: spread all colors evenly around the full circle.
        # Offset by half a step so no generated color lands exactly on the
        # chosen color's hue.
        step = 360.0 / n_colors
        for i in range(n_colors):
            h_i = (hue_deg + step / 2.0 + i * step) % 360.0
            rgb_i = colorsys.hsv_to_rgb(h_i / 360.0, s, v)
            derived.append((*rgb_i, alpha))
    else:
        # Allowed arc is everything outside +/- forbidden_angle around hue_deg.
        # Points are placed evenly across that arc, starting and ending exactly
        # at its two edges (right up against the exclusion zone).
        arc = 360.0 - 2.0 * forbidden_angle
        step = arc / (n_colors - 1)
        for i in range(n_colors):
            h_i = (hue_deg + forbidden_angle + i * step) % 360.0
            rgb_i = colorsys.hsv_to_rgb(h_i / 360.0, s, v)
            derived.append((*rgb_i, alpha))

    colors = [
        (0.0, 0.0, 0.0, 0.0),  # index 0: transparent
        (r, g, b, alpha),        # index 1: chosen color
        *derived,               # index 2..: generated colors
    ]

    return ListedColormap(colors, name=name)


if __name__ == "__main__":
    # Example usage
    cmap = generate_colormap(chosen_color="#3f8efc", n_colors=5, forbidden_angle=30)
    print(cmap.name, "-", cmap.N, "entries")
    for i, c in enumerate(cmap.colors):
        print(i, [round(x, 3) for x in c])