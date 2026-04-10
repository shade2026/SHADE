import numpy as np
from scipy.optimize import minimize
from scipy.stats import genextreme


def iqr_threshold(errors, iqr_multiplier=1.5, tail='upper'):
    errors = np.asarray(errors).flatten()

    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1

    if tail == 'upper':
        threshold = Q3 + iqr_multiplier * IQR
    elif tail == 'lower':
        threshold = Q1 - iqr_multiplier * IQR
    else:  # 'both'
        upper_threshold = Q3 + iqr_multiplier * IQR
        lower_threshold = Q1 - iqr_multiplier * IQR
        threshold = upper_threshold

    return threshold


def pot_threshold(errors, risk_level=1e-4, initial_quantile=0.95):
    errors = np.asarray(errors).flatten()
    n = len(errors)

    t0 = np.quantile(errors, initial_quantile)
    exceedances = errors[errors > t0] - t0

    if len(exceedances) < 10:
        threshold = np.quantile(errors, 0.99)
        outlier_indices = np.where(errors > threshold)[0].tolist()
        return threshold, outlier_indices, {'shape': 0, 'scale': 0}

    def neg_log_likelihood(params, data):
        shape, scale = params
        if scale <= 0:
            return 1e10
        if shape != 0:
            if np.any(1 + shape * data / scale <= 0):
                return 1e10
            ll = -np.sum(np.log(1 / scale) - (1 + 1 / shape) * np.log(1 + shape * data / scale))
        else:
            ll = -np.sum(np.log(1 / scale) - data / scale)
        return ll

    result = minimize(neg_log_likelihood, x0=[0.1, np.std(exceedances)],
                      args=(exceedances,), method='L-BFGS-B',
                      bounds=[(-0.5, 0.5), (1e-6, None)])
    shape, scale = result.x

    N_t = len(exceedances)
    q = risk_level

    if shape != 0:
        threshold = t0 + (scale / shape) * ((n * q / N_t) ** (-shape) - 1)
    else:
        threshold = t0 + scale * np.log(n * q / N_t)

    threshold = max(threshold, t0)

    return threshold


def mean_std_threshold(errors, n_sigma=3):
    errors = np.asarray(errors).flatten()

    mean_val = np.mean(errors)
    std_val = np.std(errors)
    threshold = mean_val + n_sigma * std_val

    return threshold


def mad_threshold(errors, threshold_factor=3.5):
    errors = np.asarray(errors).flatten()
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    threshold = median + threshold_factor * mad
    return threshold
