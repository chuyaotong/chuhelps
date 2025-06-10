import numpy as np
from lmfit import Model
from scipy.signal import (
    correlate,
    correlation_lags,
    find_peaks,
    peak_prominences,
    peak_widths,
)
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from helpers.math_models import (
    gaussian,
    double_gaussian,
    exp_decay,
    fano,
    double_fano,
    double_fano_bgpoly1,
    lorentzian,
    avoided_crossing,
    exp_sin,
    gaussian_2d,
    n_gaussians_2d,
)


def generic_fit(
    xdata,
    ydata,
    model,
    init_params=None,
    fixed_params=None,
    linked_params=None,
    independent_vars=None,
    weights=None,
    verbose=False,
    silence=False,
    return_result=False,
):
    """
    Perform a generic curve fitting using the provided model function.

    Supports 1D and nD data via `independent_vars`.

    Parameters:
    -----------
    xdata : array-like or tuple of arrays
        Independent variable(s). For 1D: a single array. For multi-D: a tuple
        matching `independent_vars` names.
    ydata : array-like
        Dependent variable data points (flattened if multi-D).
    model : callable
        Model function to fit to the data. Its signature must accept
        arguments matching `independent_vars` followed by parameter names.
    init_params : dict, optional
    fixed_params : set, list, or dict, optional
    linked_params : dict or iterable of tuples, optional
    independent_vars : list of str, optional
        Names of the independent coordinate arguments in the model.
        e.g. ['coords'] for a single tuple argument, or ['x','y'] for two.
    weights : array-like, optional
    verbose : bool, optional
    silence : bool, optional
    return_result : bool, optional
    """

    if independent_vars:
        mod = Model(model, independent_vars=independent_vars)
    else:
        mod = Model(model)

    if verbose:
        print(f"initial guesses: {init_params}")
    params = mod.make_params(**(init_params or {}))

    if linked_params:
        if not isinstance(linked_params, dict):
            linked_params = {d: m for m, d in linked_params}
        for dep, expr in linked_params.items():
            if dep not in params:
                raise KeyError(f"No parameter named '{dep}' to link")
            params[dep].expr = str(expr)

    if fixed_params:
        if isinstance(fixed_params, dict):
            items = fixed_params.items()
        else:
            items = ((n, None) for n in fixed_params)
        for name, val in items:
            if name not in params:
                raise KeyError(f"No parameter named '{name}'")
            params[name].vary = False
            if val is not None:
                params[name].value = val

    # prepare fit kwargs for multi-D
    fit_kwargs = {}
    if independent_vars:
        if not isinstance(xdata, (tuple, list)):
            raise ValueError("xdata must be tuple/list when independent_vars is set")
        if len(independent_vars) != len(xdata):
            raise ValueError(
                "Number of independent_vars must match length of xdata tuple"
            )
        fit_kwargs = {name: arr for name, arr in zip(independent_vars, xdata)}
    else:
        fit_kwargs = {"x": xdata}
    if weights is not None:
        fit_kwargs["weights"] = weights

    # first fit attempt
    try:
        result = mod.fit(ydata, params, **fit_kwargs)
        if verbose:
            print(result.fit_report())
        elif not silence:
            print(quick_fit_summary(result))
    except Exception as e:
        print(f"Error during fitting: {e}")
        result = None

    if not fit_is_good(result, verbose=verbose):
        if not silence:
            print("First fit bad; retrying with ampgo (max_nfev=50000)")
        try:
            result2 = mod.fit(
                ydata, params, method="ampgo", max_nfev=50000, **fit_kwargs
            )
            if verbose:
                print(result2.fit_report())
            elif not silence:
                print(quick_fit_summary(result2))

            if fit_is_good(result2, verbose=verbose):
                result = result2
            else:
                print("AMPGO fit failed; returning to first fit")

        except Exception as e:
            print(f"Error during AMPGO retry: {e}")

    if result is not None:
        values = [p.value for p in result.params.values()]
        stds = [p.stderr for p in result.params.values()]
    else:
        values = [spec["value"] for spec in init_params.values()]
        stds = np.full(len(values), np.nan)
        print("Fits failed; returning initial guesses with NaN stds")

    return result if return_result else (values, stds)


# -------------------------------------------------------------------
# Fit related helper functions
# -------------------------------------------------------------------


def fit_is_good(result, redchi_max=10, rel_err_max=1, atol=1e-12, verbose=False):

    if result is None:
        return False
    if result.weights is not None and np.ptp(result.weights) > 0:
        if (not np.isfinite(result.redchi)) or result.redchi > redchi_max:
            return False

    for p in result.params.values():
        if not p.vary:
            continue
        if p.min is not None and abs(p.value - p.min) < atol:
            print(f"Parameter {p.name} is hitting lower bound: {p.min}")
            return False
        if p.max is not None and abs(p.value - p.max) < atol:
            print(f"Parameter {p.name} is hitting upper bound: {p.max}")
            return False
        if p.stderr is None or p.stderr > abs(p.value) * rel_err_max:
            print(f"Parameter {p.name} has large stderr: {p.stderr}")
            return False

    return True


def update_params(init_params, custom_params):
    """
    Helper. Update parameter bounds in the initial parameters dictionary.

    Parameters:
    -----------
    init_params : dict
        Dictionary of initial parameter values and bounds
    custom_params : dict
        Dictionary of user-provided parameter bounds to override defaults

    Returns:
    --------
    dict
        Updated initial parameters dictionary
    """
    if not custom_params:
        return init_params

    updated_params = init_params.copy()

    for param_name, bounds in custom_params.items():
        if param_name in updated_params:

            if "min" in bounds:
                updated_params[param_name]["min"] = bounds["min"]
            if "max" in bounds:
                updated_params[param_name]["max"] = bounds["max"]
            if "value" in bounds:
                updated_params[param_name]["value"] = bounds["value"]

    return updated_params


def quick_fit_summary(result):
    """
    Helper. Format the fit results into quick summary string with params and stds.

    Parameters:
    -----------
    result : lmfit.model.ModelResult
        The result object from the model fitting

    Returns:
    --------
    str
        A formatted string containing parameter names, values, and standard errors
    """
    summary = "  ".join(
        f"{name} = {p.value:.4f} ± {(p.stderr or np.nan):.4f}"
        for name, p in result.params.items()
    )
    return summary


def find_2d_clusters_kmeans(
    n,
    coords,
    weights=None,
    random_state=0,
    n_init=10,
    return_clusters=True,
):
    """
    Finds clusters using KMeans and returns a dictionary where keys are cluster IDs
    and values are dictionaries containing the cluster's center, points, and point_weights.
     Parameters
    ----------
    n : int
        The number of clusters to form (and the number of centroids to generate).
    coords : numpy.ndarray
        A 2D array of shape (n_samples, 2) containing the (x, y) coordinates
        of the data points to be clustered.
    weights : Optional[numpy.ndarray], default=None
        A 1D array of shape (n_samples,) representing the weights for each data
        point in `coords`. If None, all points are assigned a weight of 1.
    random_state : int, default=0
        Determines random number generation for centroid initialization. Use an int
        for reproducible results.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    return_clusters : bool, default=True
        If True, the function returns a dictionary where each key is a cluster ID
        and the value is a dictionary containing the 'center', 'points', and
        'point_weights' for that cluster.
        If False, the function returns a tuple containing the global
        `cluster_centers`, point `labels`, and the processed `weights` array.

    Returns
    -------
    Dict[int, Dict[str, np.ndarray]] or Tuple[np.ndarray, np.ndarray, np.ndarray]
        The return type depends on the `return_clusters` flag:

        If `return_clusters` is True:
            A dictionary where keys are integer cluster IDs (from 0 to `n`-1)
            and values are dictionaries with the following structure:
            - 'center' (np.ndarray): The (x, y) coordinates of the cluster center.
                                     Shape: (2,).
            - 'points' (np.ndarray): An array of (x, y) coordinates of the points
                                     belonging to this cluster. Shape: (n_points_in_cluster, 2).
            - 'point_weights' (np.ndarray): An array of weights for the points
                                            in this cluster. Shape: (n_points_in_cluster,).

        If `return_clusters` is False:
            A tuple `(cluster_centers, labels, weights)`:
            - cluster_centers (np.ndarray): Coordinates of cluster centers.
                                            Shape: (`n`, 2).
            - labels (np.ndarray): Label assigned to each point in `coords`.
                                   Shape: (n_samples,). Values range from 0 to `n`-1.
            - weights (np.ndarray): The processed weights array used for fitting.
                                    Shape: (n_samples,). (This is the input `weights`
                                    or an array of ones if `weights` was None).
    """

    if weights is None:
        weights = np.ones(len(coords))

    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=n_init)
    kmeans.fit(coords, sample_weight=weights)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    if return_clusters:
        clusters = {}

        for i in range(n):

            cluster_mask = labels == i
            cluster_points = coords[cluster_mask]
            cluster_weights = weights[cluster_mask]

            clusters[i] = {
                "center": cluster_centers[i],
                "points": cluster_points,
                "point_weights": cluster_weights,
            }

        return clusters
    else:
        return cluster_centers, labels, weights


def get_cluster_shape(
    points,
    weights,
    x_min,
    x_max,
    y_min,
    y_max,
    num_clusters,
):
    """
    Estimates sigma_x, sigma_y, and theta from a cluster's points and weights,
    using global grid dimensions for fallback and floor values.

    Parameters
    ----------
    points : numpy.ndarray
        An Nx2 array of (x, y) coordinates representing the points within the
        specific cluster.
    weights : numpy.ndarray
        A 1D array of N weights, corresponding to each point in the `points`
        array.
    x_min : float
        The minimum x-coordinate of the global grid context. Used for calculating
        fallback sigma values if the primary estimation fails.
    x_max : float
        The maximum x-coordinate of the global grid context. Used for calculating
        fallback sigma values.
    y_min : float
        The minimum y-coordinate of the global grid context. Used for calculating
        fallback sigma values.
    y_max : float
        The maximum y-coordinate of the global grid context. Used for calculating
        fallback sigma values.
    num_clusters : int
        The total number of clusters being considered in the larger analysis.
        This is used to scale the fallback sigma values.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the estimated shape parameters:
        - sigma_x (float): Estimated standard deviation along the Gaussian's
                           first principal axis.
        - sigma_y (float): Estimated standard deviation along the Gaussian's
                           second principal axis.
        - theta (float): Estimated rotation angle in radians of the Gaussian's
                         first principal axis relative to the positive x-axis.
                         Defaults to 0.0 in fallback cases or if variances
                         do not support orientation calculation.
    """
    try:

        total_wt = weights.sum()
        cluster_points_mean = np.average(points, axis=0, weights=weights)

        cov = np.zeros((2, 2))
        for p, wt in zip(points, weights):
            d = p - cluster_points_mean
            cov += wt * np.outer(d, d)

        cov /= total_wt
        var_x = cov[0, 0]
        var_y = cov[1, 1]

        sigma_x = np.sqrt(var_x) if var_x > 0 else 0.0
        sigma_y = np.sqrt(var_y) if var_y > 0 else 0.0

        if var_x > 0 and var_y > 0:
            # Avoid issues if var_x is very close to var_y and cov[0,1] is also very small
            if not (np.isclose(var_x, var_y) and np.isclose(cov[0, 1], 0.0)):
                theta = 0.5 * np.arctan2(2 * cov[0, 1], var_x - var_y)
            else:
                theta = 0.0
        else:
            theta = 0.0

    except Exception as e:
        print(f"Error estimating cluster shape parameters: {e}")

        sigma_x = (x_max - x_min) / (6 * num_clusters)
        sigma_y = (y_max - y_min) / (6 * num_clusters)
        theta = 0.0

    return sigma_x, sigma_y, theta


# -------------------------------------------------------------------
# Fit wrappers for each model. Mainly on ini_params definitions.
# -------------------------------------------------------------------


def fit_exp_decay(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):

    init_params = {
        "A": dict(value=abs((np.max(ydata) - np.min(ydata)) / 2), min=0),
        "tau": dict(value=np.max(xdata) / 2, min=0),
        "C": dict(value=np.mean(ydata), min=0),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        exp_decay,
        init_params=init_params,
        **kwargs,
    )


def fit_exp_sin(xdata, ydata, custom_params=None, **kwargs):

    if ydata[0] > np.mean(ydata):
        phase_guess = np.pi / 2
    else:
        phase_guess = -np.pi / 2

    init_params = {
        "A": dict(value=abs((np.max(ydata) - np.min(ydata))), min=np.std(ydata) / 5),
        "tau": dict(value=np.max(xdata) / 2, min=abs(xdata[1] - xdata[0])),
        "freq": dict(value=2 * np.pi * 5 / np.max(xdata), min=0),
        "phi": dict(value=phase_guess, min=-np.pi, max=np.pi),
        "C": dict(value=np.mean(ydata), min=0),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        exp_sin,
        init_params=init_params,
        **kwargs,
    )


def fit_lorentzian(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):

    C_guess = np.median(ydata)

    peak_idx_guess = np.argmax(np.abs(ydata - C_guess))

    x0_guess = xdata[peak_idx_guess]
    C_guess = np.median(ydata)
    A_guess = ydata[peak_idx_guess] - C_guess

    gamma_guess = 4 * (xdata[1] - xdata[0])  # assuming we sampled the peak well

    init_params = {
        "x0": dict(value=x0_guess, min=np.min(xdata), max=np.max(xdata)),
        "gamma": dict(value=gamma_guess, min=0, max=xdata[-1] - xdata[0]),
        "A": dict(value=A_guess),
        "C": dict(value=C_guess, min=min(ydata), max=max(ydata)),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        lorentzian,
        init_params=init_params,
        **kwargs,
    )


def fit_gaussian(
    xdata=None,
    ydata=None,
    data=None,  # can auto bin a distribution data
    bins=80,
    custom_params=None,
    return_data=False,  # ... and return binned data
    **kwargs,
):
    if xdata is None and ydata is None and data is not None:
        counts, bins = np.histogram(data, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ydata = counts
        xdata = bin_centers

        sigma_guess = np.std(data)
        mu_guess = np.mean(data)
    else:
        sigma_guess = (np.max(xdata) - np.min(xdata)) / 10
        mu_guess = xdata[np.argmax(ydata)]

    A_guess = np.max(ydata)
    C_guess = np.mean(ydata[0], ydata[-1])

    init_params = {
        "mu": dict(value=mu_guess, min=np.min(xdata), max=np.max(xdata)),
        "sigma": dict(value=sigma_guess, min=0),
        "A": dict(value=A_guess, min=0),
        "C": dict(value=C_guess),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    result = generic_fit(
        xdata,
        ydata,
        gaussian,
        init_params=init_params,
        **kwargs,
    )

    if return_data:
        return (result, (xdata, ydata))
    else:
        return result


def fit_double_gaussian(
    xdata=None,
    ydata=None,
    data=None,  # can auto bin a distribution data
    bins=80,
    smooth=3,
    custom_params=None,
    return_data=False,  # ... and return binned data
    **kwargs,
):

    if xdata is None and ydata is None and data is not None:
        counts, bins = np.histogram(data, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ydata = counts
        xdata = bin_centers

    ydata = median_filter(ydata, size=smooth)  # smoothing data

    peaks, _ = find_peaks(ydata, distance=len(xdata) // 5)  # ≥5 % apart
    prom = peak_prominences(ydata, peaks)[0]

    if len(peaks) >= 2:
        pk1, pk2 = peaks[prom.argsort()[-2:]][::-1]
    else:
        pk1 = peaks[np.argmax(prom)] if len(peaks) else np.argmax(ydata)
        pk2 = pk1

    C_guess = 0

    mu1_guess = xdata[pk1]
    mu2_guess = xdata[pk2]

    A1_guess = ydata[pk1]
    A2_guess = ydata[pk2]

    w1 = peak_widths(ydata, [pk1], rel_height=0.5)[0][0] * (xdata[1] - xdata[0])
    w2 = peak_widths(ydata, [pk2], rel_height=0.5)[0][0] * (xdata[1] - xdata[0])
    sigma1_guess = w1 / (2 * np.sqrt(2 * np.log(2)))  # σ = FWHM / 2√(2ln2)
    sigma2_guess = w2 / (2 * np.sqrt(2 * np.log(2)))

    init_params = {
        "mu1": dict(value=mu1_guess, min=np.min(xdata), max=np.max(xdata)),
        "sigma1": dict(value=sigma1_guess, min=0),
        "A1": dict(value=A1_guess, min=0, max=np.max(ydata)),
        "mu2": dict(value=mu2_guess, min=np.min(xdata), max=np.max(xdata)),
        "sigma2": dict(value=sigma2_guess, min=0),
        "A2": dict(value=A2_guess, min=0, max=np.max(ydata)),
        "C": dict(value=C_guess),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    result = generic_fit(
        xdata,
        ydata,
        double_gaussian,
        init_params=init_params,
        **kwargs,
    )

    if return_data:
        return (result, (xdata, ydata))
    else:
        return result


def fit_fano(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):

    xspan = np.max(xdata) - np.min(xdata)
    yspan = np.max(ydata) - np.min(ydata)

    f_guess = np.mean(xdata)

    A_guess = yspan / 2

    kappa_guess = xspan / 5
    q_guess = 1

    C_guess = np.min(ydata)
    noise_est = np.std(ydata[:5]) / 10

    init_params = {
        "f": dict(value=f_guess, min=np.min(xdata), max=np.max(xdata)),
        "kappa": dict(
            value=kappa_guess,
            min=10 * (xdata[1] - xdata[0]),
            max=max(xdata) - min(xdata),
        ),
        "q": dict(value=q_guess),
        "A": dict(value=A_guess, min=min(ydata)),
        "C": dict(value=C_guess, min=noise_est, max=max(ydata)),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        fano,
        init_params=init_params,
        **kwargs,
    )


def fit_double_fano(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):
    xspan = np.max(xdata) - np.min(xdata)
    yspan = np.max(ydata) - np.min(ydata)

    f_guess = np.mean(xdata)

    A_guess = yspan / 2

    kappa_guess = xspan / 5
    q_guess = 1
    chi_guess = kappa_guess / 5

    C_guess = np.min(ydata)

    noise_est = np.std(ydata[:5]) / 10

    init_params = {
        "f": dict(value=f_guess, min=np.min(xdata), max=np.max(xdata)),
        "kappa": dict(
            value=kappa_guess,
            min=10 * (xdata[1] - xdata[0]),
            max=max(xdata) - min(xdata),
        ),
        "q": dict(value=q_guess, min=-2, max=2),
        "A": dict(value=A_guess, min=min(ydata)),
        "chi": dict(
            value=chi_guess, min=xdata[1] - xdata[0], max=max(xdata) - min(xdata)
        ),
        "C": dict(value=C_guess, min=noise_est, max=max(ydata)),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        double_fano,
        init_params=init_params,
        **kwargs,
    )


def fit_double_fano_bgpoly1(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):
    xspan = np.max(xdata) - np.min(xdata)
    yspan = np.max(ydata) - np.min(ydata)

    f_guess = np.mean(xdata)

    A_guess = yspan / 2

    kappa_guess = xspan / 10
    q_guess = 1
    chi_guess = kappa_guess

    C_guess = np.min(ydata)
    slope_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

    init_params = {
        "f": dict(value=f_guess, min=np.min(xdata), max=np.max(xdata)),
        "kappa": dict(value=kappa_guess, min=0, max=max(xdata) - min(xdata)),
        "q": dict(value=q_guess),
        "A": dict(value=A_guess, min=0),
        "chi": dict(value=chi_guess, min=0, max=max(xdata) - min(xdata)),
        "C": dict(value=C_guess, min=0, max=max(ydata)),
        "slope": dict(value=slope_guess),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        double_fano_bgpoly1,
        init_params=init_params,
        **kwargs,
    )


def fit_avoided_crossing(
    xdata,
    ydata,
    custom_params=None,
    **kwargs,
):

    x0_guess = np.mean(xdata)

    omega_guess = 1 / (np.mean(ydata) * 4)

    amplitude_guess = 1

    init_params = {
        "x0": dict(value=x0_guess),
        "omega": dict(value=omega_guess, min=0),
        "A": dict(value=amplitude_guess, min=0),
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    return generic_fit(
        xdata,
        ydata,
        avoided_crossing,
        init_params=init_params,
        **kwargs,
    )


def fit_gaussian_2d(
    xdata,
    ydata,
    coords=None,
    bins=100,
    smooth=1,
    symmetric=False,
    custom_params=None,
    fixed_params={},
    linked_params=set(),
    **kwargs,
):

    counts, xedges, yedges = np.histogram2d(xdata, ydata, bins=bins)
    z_image = counts.T
    x_axis = 0.5 * (xedges[:-1] + xedges[1:])
    y_axis = 0.5 * (yedges[:-1] + yedges[1:])

    Z = median_filter(z_image, size=smooth) if smooth else z_image.copy()
    X, Y = np.meshgrid(x_axis, y_axis)
    coords_hist = (X.ravel(), Y.ravel())
    z_vec = Z.ravel()

    ij = np.unravel_index(np.argmax(Z), Z.shape)
    A0 = Z[ij] - np.median(Z)
    x0, y0 = X[ij], Y[ij]
    sigx0 = (x_axis[-1] - x_axis[0]) / 6
    sigy0 = (y_axis[-1] - y_axis[0]) / 6
    C0 = 0

    init_params = {
        "A": {"value": max(A0, 1e-3), "min": 0},
        "x0": {"value": x0, "min": x_axis.min(), "max": x_axis.max()},
        "y0": {"value": y0, "min": y_axis.min(), "max": y_axis.max()},
        "sigma_x": {"value": sigx0, "min": 1e-3},
        "sigma_y": {"value": sigy0, "min": 1e-3},
        "theta": {"value": 0.0},
        "C": {"value": C0},
    }

    if custom_params:
        init_params = update_params(init_params, custom_params)

    if symmetric:
        linked_params.add(("sigma_x", "sigma_y"))
        fixed_params["theta"] = 0

    result = generic_fit(
        xdata=coords_hist,
        ydata=z_vec,
        model=gaussian_2d,
        init_params=init_params,
        independent_vars=["x", "y"],
        fixed_params=fixed_params,
        linked_params=linked_params,
        **kwargs,
    )

    return result


def fit_n_gaussians_2d(
    n,
    xdata=None,
    ydata=None,
    coords=None,
    bins=100,
    smooth=1,
    symmetric=False,
    coord_weights=None,
    custom_params=None,
    fixed_params={},
    linked_params=set(),
    **kwargs,
):

    if coords is None:
        if xdata is not None and ydata is not None:
            coords = np.column_stack((xdata, ydata))
        else:
            raise ValueError("Either coords, or xdata and ydata must be provided.")

    counts, x_edges, y_edges = np.histogram2d(xdata, ydata, bins=bins)

    Z = median_filter(counts.T, size=smooth)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)
    coords_hist = (X.ravel(), Y.ravel())
    z_vec = Z.ravel()

    x_min, x_max = x_centers.min(), x_centers.max()
    y_min, y_max = y_centers.min(), y_centers.max()
    Z_min = Z.min() if Z.size > 0 else 0.0

    init_params = {}
    init_params["C"] = {"value": Z.min()}

    clusters = find_2d_clusters_kmeans(
        n, coords, weights=coord_weights, return_clusters=True
    )

    for cluster_id, cluster_data in clusters.items():

        x0, y0 = cluster_data["center"]

        points_for_shape = cluster_data["points"]
        weights_for_shape = cluster_data["point_weights"]

        sigma_x, sigma_y, theta = get_cluster_shape(
            points_for_shape, weights_for_shape, x_min, x_max, y_min, y_max, n
        )

        ix = np.argmin(np.abs(x_centers - x0))
        iy = np.argmin(np.abs(y_centers - y0))

        A0_guess = Z[iy, ix] - Z_min

        param_idx = cluster_id + 1
        init_params[f"A{param_idx}"] = {"value": A0_guess, "min": 1e-3}
        init_params[f"x0_{param_idx}"] = {
            "value": x0,
            "min": x_min,
            "max": x_max,
        }
        init_params[f"y0_{param_idx}"] = {
            "value": y0,
            "min": y_min,
            "max": y_max,
        }
        init_params[f"sigma_x{param_idx}"] = {
            "value": sigma_x,
            "min": x_centers[1] - x_centers[0],
        }
        init_params[f"sigma_y{param_idx}"] = {
            "value": sigma_y,
            "min": y_centers[1] - y_centers[0],
        }
        init_params[f"theta{param_idx}"] = {"value": theta}

    if custom_params:
        init_params = update_params(init_params, custom_params)

    if symmetric:
        for i in range(1, n + 1):
            if f"theta{i}" not in fixed_params:
                fixed_params[f"theta{i}"] = 0
            if (f"sigma_x{i}", f"sigma_y{i}") not in linked_params:
                linked_params.add((f"sigma_x{i}", f"sigma_y{i}"))

    result = generic_fit(
        xdata=coords_hist,
        ydata=z_vec,
        model=n_gaussians_2d(n),
        init_params=init_params,
        independent_vars=["x", "y"],
        fixed_params=fixed_params,
        linked_params=linked_params,
        **kwargs,
    )

    return result


# -------------------------------------------------------------------
# Others data proceesing functions
# -------------------------------------------------------------------


def find_periodicity_1d(
    xdata=None,
    ydata=None,
    fit=True,
    height_thresh=0.1,
    min_lag=10,
    max_lag=-1,
    min_peak_dist=10,
):

    if ydata is None:  # assume users input their values in xdata
        if xdata is not None:
            ydata = xdata
            xdata = np.arange(len(ydata))
        else:
            raise ValueError(
                "Either xdata or ydata, at least one 1d data must be provided."
            )
    else:
        if xdata is None:
            xdata = np.arange(len(ydata))

    N = len(xdata)

    signal_std = np.std(ydata)
    signal_mean = np.mean(ydata)

    norm_signal = (ydata - signal_mean) / signal_std
    autocorr = correlate(norm_signal, norm_signal, mode="full")
    autocorr = autocorr[N - 1 :]
    autocorr /= autocorr[0]

    stds = np.full(5, np.nan)

    if fit:
        custom_parmas = {
            "C": {"value": 0, "min": -2, "max": 2},
        }
        fixed_params = {
            "phi": np.pi / 2,  # fix C to 0
            # "C": 0,  # fix C to 0
        }
        params, stds = fit_exp_sin(
            xdata,
            autocorr,
            custom_params=custom_parmas,
            fixed_params=fixed_params,
            verbose=False,
        )

    if not np.all(np.isnan(stds)):
        period = 2 * np.pi / params[2]
        return period, autocorr

    else:
        print("Decaying sinosoidal fit failed or turned off; using peaks find")
        peaks_idx, properties = find_peaks(
            autocorr[min_lag : max_lag + 1],
            height=height_thresh,
            distance=min_peak_dist,
        )
        if peaks_idx.size == 0:
            print("No peaks found in the autocorrelation signal.")
            return None, autocorr
        else:
            first_peak_idx = peaks_idx[0]

            if first_peak_idx >= 1:
                avg_x_step = (xdata[-1] - xdata[0]) / (N - 1)
                period_in_x = first_peak_idx * avg_x_step

                return period_in_x, autocorr


def find_symmetry_centers_1d(
    xdata=None, ydata=None, peak_height_thresh=10, min_peak_dist=5, verbose=False
):

    if ydata is None:  # assume users input their values in xdata
        if xdata is not None:
            ydata = xdata
            xdata = np.arange(len(ydata))
            xlabel = "index"
        else:
            raise ValueError(
                "Either xdata or ydata, at least one 1d data must be provided."
            )
    else:
        if xdata is None:
            xdata = np.arange(len(ydata))
            xlabel = "index"
        else:
            xlabel = "x"

    N = len(ydata)

    signal_std = np.std(ydata)
    signal_mean = np.mean(ydata)

    y_norm = y_norm = (ydata - signal_mean) / signal_std

    y_norm_flipped = np.flip(y_norm)

    corr = correlate(y_norm, y_norm_flipped, mode="full", method="auto")
    lags = correlation_lags(len(y_norm), len(y_norm_flipped), mode="full")

    norm_corr = corr / corr[0]

    peak_indices, properties = find_peaks(
        norm_corr, height=peak_height_thresh, distance=min_peak_dist
    )

    symmetry_centers = []
    for peak_idx in peak_indices:
        peak_lag = lags[peak_idx]
        peak_height = norm_corr[peak_idx]

        center_xidx = (N - 1 + peak_lag) / 2.0

        center_x = np.interp(center_xidx, np.arange(N), xdata)
        symmetry_centers.append(
            {
                "x": center_x,
                "height": peak_height,
            }
        )

    sorted_centers = sorted(
        symmetry_centers, key=lambda item: item["height"], reverse=True
    )

    sorted_centers_x = []
    for item in sorted_centers:
        sorted_centers_x.append(item["x"])
        if verbose:
            print(
                f"Symmetry center at {xlabel} = {item['x']:.4f} (Height: {item['height']:.2f})"
            )

    return sorted_centers_x
