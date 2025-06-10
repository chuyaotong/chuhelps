import numpy as np
from math import factorial
from scipy.special import factorial2


def sigma_from_flux_noise(flux, f01s, order=1, A_phi=20e-6, diagonal_only=True):

    var = np.zeros_like(flux, dtype=float)

    # -- diagonal (k = k) contributions
    for k in range(1, order):
        df01dphi = df_dflux(f01s, flux, order=k)
        coeff = df01dphi / factorial(k)
        moment = factorial2(2 * k - 1) * A_phi ** (2 * k)  # (2k-1)!! σ^{2k}
        var += coeff**2 * moment

    # # -- optional cross-terms  (disabled by default)
    # if not diagonal_only:
    #     for k in range(1, max_order + 1):
    #         dk = np.asarray(derivs[k - 1], dtype=float) / factorial(k)
    #         for m in range(k + 1, max_order + 1):
    #             dm = np.asarray(derivs[m - 1], dtype=float) / factorial(m)
    #             order = k + m
    #             # ⟨δΦ^{order}⟩  (only even moments survive)
    #             if order % 2 == 0:
    #                 moment = factorial2(order - 1) * A_phi**order
    #                 var += 2 * dk * dm * moment  # factor 2 for symmetry

    sigma_E = np.sqrt(var)
    return sigma_E


def df_dflux(f, flux, order=1):

    df01dflux = f.copy()
    for i in range(order):
        # df01dflux_current = df01dflux.copy()
        df01dflux = np.gradient(df01dflux, flux)

    return df01dflux
