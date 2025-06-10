import numpy as np
from matplotlib.patches import Ellipse


def gaussian(x, mu, sigma, A, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + C


def exp_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C


def sine(x, A, freq, phi, C):
    return A * np.sin(freq * x + phi) + C


def poly1(x, a, C):
    return a * x + C


def poly2(x, a, b, C):
    return a * x**2 + b * x + C


def fano(x, f, kappa, q, A, C):
    return A * (q * kappa / 2 + x - f) ** 2 / ((kappa / 2) ** 2 + (x - f) ** 2) + C


def lorentzian(x, x0, gamma, A, C):
    return A * ((gamma / 2) / ((x - x0) ** 2 + (gamma / 2) ** 2)) + C


def avoided_crossing(x, x0, omega, A):
    delta = x - x0
    return A / np.sqrt(omega**2 + delta**2)


def gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y, theta, C):
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    expo = (xr**2) / (2 * sigma_x**2) + (yr**2) / (2 * sigma_y**2)
    return A * np.exp(-expo) + C


# -------------------------------------------------------------------
# Composite models
# -------------------------------------------------------------------


def exp_sin(x, A, tau, freq, phi, C):
    exp = exp_decay(x, A, tau, C=0)
    sin = sine(x, A, freq, phi, C=0)
    return exp * sin + C


def double_fano(x, f, chi, kappa, q, A, C):
    fano1 = fano(x, f - chi / 2, kappa, q, A, C=0)
    fano2 = fano(x, f + chi / 2, kappa, q, A, C=0)
    return fano1 + fano2 + C


def fano_bgpoly1(x, f0, kappa, q, A, offset, slope):
    fano_val = fano(x, f0, kappa, q, A, C=0)
    bg = poly1(x, slope, offset)
    return fano_val + bg


def double_fano_bgpoly1(x, f, kappa, q, A, chi, C, slope):
    double_fano_val = double_fano(x, f, chi, kappa, q, A, C=0)
    bg = poly1(x, slope, C)
    return double_fano_val + bg


def double_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2, C):
    gaussian1 = gaussian(x, mu1, sigma1, A1, 0)
    gaussian2 = gaussian(x, mu2, sigma2, A2, 0)
    return gaussian1 + gaussian2 + C


def n_gaussians_2d(n):

    param_names = ["x", "y", "C"]
    for i in range(1, n + 1):
        param_names.extend(
            [f"A{i}", f"x0_{i}", f"y0_{i}", f"sigma_x{i}", f"sigma_y{i}", f"theta{i}"]
        )

    func_def = f"def model({', '.join(param_names)}):\n"
    func_def += "    z = C * np.ones_like(x)\n"
    for i in range(1, n + 1):
        func_def += f"    z += gaussian_2d(x, y, A{i}, x0_{i}, y0_{i}, sigma_x{i}, sigma_y{i}, theta{i}, 0)\n"
    func_def += "    return z"

    namespace = {"gaussian_2d": gaussian_2d, "np": np}
    exec(func_def, namespace)
    return namespace["model"]


# -------------------------------------------------------------------
# Others
#  -------------------------------------------------------------------


def gaussian_2d_ellipse(
    x0,
    y0,
    sigma_x,
    sigma_y,
    theta,
    nsigma=1,
    facecolor="none",
    edgecolor="k",
    lw=1,
    ls="--",
    alpha=0.5,
    **kwargs,
):

    return Ellipse(
        (x0, y0),
        width=2 * sigma_x * nsigma,
        height=2 * sigma_y * nsigma,
        angle=theta,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=lw,
        ls="--",
        alpha=0.5,
        **kwargs,
    )
