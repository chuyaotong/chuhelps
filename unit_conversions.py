import numpy as np
from scipy.constants import h, hbar, e, pi

Phi_0 = h / (2 * e)
red_Phi_0 = hbar / (2 * e)


def losc_from_phizpf(phi_zpf):
    return phi_zpf * (np.sqrt(2))


def phizpf_from_Zmode(Zmode):
    """
    Calculate zero-point fluctuation of flux [Phi_0 / 2pi] for a resonator impedance [Ohm].

    Parameters
    ----------
    Z : float
        Resonator impedance [ohm].

    Returns
    -------
    float
        Zero-point fluctuation of flux [Phi_0 / 2pi].
    """
    Phi_zpf = np.sqrt(hbar * Zmode / 2)
    return Phi_zpf / red_Phi_0


def phizpf_from_Z0(Z0, ratio=4):
    """
    Calculate zero-point fluctuation of flux [Phi_0 / 2pi] for a resonator impedance [Ohm].

    Parameters
    ----------
    Z : float
        Resonator impedance [ohm].

    Returns
    -------
    float
        Zero-point fluctuation of flux [Phi_0 / 2pi].
    """
    Zmode = Z0 * ratio / pi
    Phi_zpf = np.sqrt(hbar * Zmode / 2)
    return Phi_zpf / red_Phi_0


def phizpf_from_C_fres(Cres, fres):
    """
    Calculate zero-point fluctuation of flux [Phi_0 / 2pi] for a resonator capacitance [fF] and frequency [GHz].

    Parameters
    ----------
    Cres : float
        Resonator capacitance [fF].
    fres : float
        Resonator frequency [GHz].

    Returns
    -------
    float
        Zero-point fluctuation of flux [Phi_0 / 2pi].
    """
    wres = 2 * pi * fres * 1e9
    Phi_zpf = np.sqrt(hbar / (2 * Cres * 1e-15 * wres))

    return Phi_zpf / red_Phi_0


def Cres_from_fres_Z0(fres, Z0=50, ratio=4):
    """
    Lumped element. Calculate resonator capacitance [fF] from frequency [GHz].

    Parameters
    ----------
    fres : float
        Resonator frequency [GHz].
    Z : float, optional
        Impedance [ohms], by default 50.
    ratio : float, optional
        For lambda over ratio resonator, by default 4.

    Returns
    -------
    float
        Resonator capacitance [fF].
    """
    wres = 2 * pi * fres * 1e9
    Zmode = Z0 * ratio / pi
    return 1 / (wres * Zmode) / 1e-15


def Lres_from_fres_Z0(fres, Z0=50, ratio=4):
    """
    Lumped element. Calculate resonator inductance [uH] from frequency [GHz] and impedance [Ohm].

    Parameters
    ----------
    fres : float
        Resonator frequency [GHz].
    Z : float, optional
        Impedance [ohms], by default 50.
    ratio : float, optional
        For lambda over ratio resonator, by default 4.

    Returns
    -------
    float
        Resonator inductance [uH].
    """
    Zmode = Z0 * ratio / pi
    wres = 2 * pi * fres * 1e9

    return (Zmode) / (wres) / 1e-6


def Lres_from_fres_Cres(fres, Cres, ratio=4):
    """
    Lumped element. Calculate resonator inductance [uH] from frequency [GHz] and capacitance [fF].

    Parameters
    ----------
    fres : float
        Resonator frequency [GHz].
    Cres : float
        Resonator capacitance [fF].
    ratio : float, optional
        For lambda over ratio resonator, by default 4.

    Returns
    -------
    float
        Resonator inductance [uH].
    """

    wres = 2 * pi * fres * 1e9
    return 1 / (wres**2 * Cres * 1e-15) / 1e-6


def fres_from_Lres_Cres(Lres, Cres):
    """
    Lumped element. Calculate resonator frequency [GHz] from inductance [uH] and capacitance [fF].

    Parameters
    ----------
    Lres : float
        Resonator inductance [uH].
    Cres : float
        Resonator capacitance [fF].

    Returns
    -------
    float
        Resonator frequency [GHz].
    """

    return 1 / (2 * pi * np.sqrt(Lres * 1e-6 * Cres * 1e-15)) / 1e9


def Zmode_from_Ltot_Ceff(Ltot, Ceff):

    return np.sqrt((Ltot * 1e-6) / (Ceff * 1e-15))


def Z0_from_fres_Cres(fres, Cres, ratio=4):
    """
    Lumped element. Calculate resonator impedance [Ohm] from frequency [GHz] and capacitance [fF].

    Parameters
    ----------
    fres : float
        Resonator frequency [GHz].
    Cres : float
        Resonator capacitance [fF].

    Returns
    -------
    float
        Resonator impedance [Ohm].
    """

    return 1 / (2 * ratio * fres * 1e9 * Cres * 1e-15)


def Zmode_from_fres_Cres(fres, Cres, ratio=4):

    Zmode = Z0_from_fres_Cres(fres, Cres, ratio)

    return Zmode * ratio / pi


def Cj_from_Aj(lj=None, wj=None, Aj=None, C_per_A=48.0):
    """
    Calculate junction capacitance [fF] from junction area [um^2].

    Parameters
    ----------
    lj : float, optional
        Junction length [um], by default None.
    wj : float, optional
        Junction width [um], by default None.
    Aj : float, optional
        Junction area [um^2], by default None.
    C_per_A : float, optional
        Capacitance per area [fF/um^2], by default 48 (LL for 0.2uA/um^2 Jc) .

    Returns
    -------
    float
        Junction capacitance [fF].

    Raises
    ------
    ValueError
        If neither Aj nor both lj and wj are provided.
    """

    if Aj is None:
        if lj is not None and wj is not None:
            Aj = lj * wj
        else:
            raise ValueError(
                "Area is not defined. Provide either Aj directly or both lj and wj."
            )

    return Aj * C_per_A


def EC_from_C(C):
    """
    Convert capacitance [fF] to charging energy [GHz].

    Parameters
    ----------
    C : float
        Capacitance [fF].

    Returns
    -------
    float
        Charging energy [GHz].
    """
    return e**2 / (2 * C * 1e-15) / h / 1e9


def C_from_EC(EC):
    """
    Convert charging energy [GHz] to capacitance [fF].

    Parameters
    ----------
    EC : float
        Charging energy [GHz].

    Returns
    -------
    float
        Capacitance [fF].
    """
    return e**2 / (2 * h * EC * 1e9) / 1e-15


def L_from_EL(EL):
    """
    Convert inductive energy [GHz] to inductance [uH].

    Parameters
    ----------
    EL : float
        Inductive energy [GHz].

    Returns
    -------
    float
        Inductance [uH].
    """

    return Phi_0**2 / (2 * pi) ** 2 / (h * EL * 1e9) / 1e-6


def EL_from_L(L):
    """
    Convert inductance [uH] to inductive energy [GHz].

    Parameters
    ----------
    L : float
        Inductance [uH].

    Returns
    -------
    float
        Inductive energy [GHz].
    """
    return Phi_0**2 / (2 * pi) ** 2 / (L * 1e-6) / h / 1e9


def flux_from_bias(bias, period, offset=None, half_flux=None, full_flux=None):
    """
    Convert physical bias [V or A] to flux [Phi_0].

    Parameters
    ----------
    bias : float
        Bias [V or A].
    period : float
        Period [V or A].
    offset : float
        Offset [V or A].

    Returns
    -------
    float
        Flux [Phi_0].
    """
    if offset is None:
        if half_flux is not None:
            offset = half_flux - period / 2
        elif full_flux is not None:
            offset = full_flux - period
        else:
            raise ValueError(
                "Offset is not defined. Provide either offset directly or half_flux or full_flux."
            )

    return (bias - offset) / period


def EJ_from_Jc_A(Jc, lj=None, wj=None, Aj=None):
    """
    Convert critical current density [uA/um^2] to Josephson energy [GHz].

    Parameters
    ----------
    Jc : float
        Critical current density [uA/um^2].
    lj : float, optional
        Junction length [um], by default None.
    wj : float, optional
        Junction width [um], by default None.
    Aj : float, optional
        Junction area [um^2], by default None.

    Returns
    -------
    float
        Josephson energy [GHz].
    """

    if Aj is None:
        if lj is not None and wj is not None:
            Aj = lj * wj
        else:
            raise ValueError(
                "Area is not defined. Provide either Aj directly or both lj and wj."
            )

    return hbar * Jc * 1e-6 * Aj / (2 * e) / 1e9 / h


def L_from_Jc_A(Jc, lj=None, wj=None, Aj=None):
    """
    Convert critical current density [uA/um^2] to inductance [uH].

    Parameters
    ----------
    Jc : float
        Critical current density [uA/um^2].
    lj : float, optional
        Junction length [um], by default None.
    wj : float, optional
        Junction width [um], by default None.
    Aj : float, optional
        Junction area [um^2], by default None.

    Returns
    -------
    float
        Inductance [uH].
    """

    if Aj is None:
        if lj is not None and wj is not None:
            Aj = lj * wj
        else:
            raise ValueError(
                "Area is not defined. Provide either Aj directly or both lj and wj."
            )

    return red_Phi_0 / (Jc * 1e-6 * Aj) / 1e-6


def fplasm_from_EJ_EC(EJ, EC):
    """
    Calculate plasma frequency from Josephson energy and charging energy.

    Parameters
    ----------
    EJ : float
        Josephson energy [any].
    EC : float
        Charging energy [any, same as above].

    Returns
    -------
    float
        Plasma frequency [same as EJ and EC].
    """
    return np.sqrt(8 * EJ * EC)
