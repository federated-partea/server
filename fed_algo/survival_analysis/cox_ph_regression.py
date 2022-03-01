import logging
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from autograd import elementwise_grad
from autograd import numpy as anp
from lifelines import exceptions
from lifelines.exceptions import ConvergenceError
from lifelines.utils import inv_normal_cdf
from scipy import stats
from scipy.linalg import solve as spsolve, LinAlgError, norm

logger = logging.getLogger(__name__)
CONVERGENCE_DOCS = "Please see the following tips in the lifelines documentation:https://lifelines.readthedocs.io"


def compute_global_mean(client_data):
    logger.info("Compute global mean")
    mean = pd.Series(dtype="float64")
    n_samples = 0
    for client_id in client_data.keys():
        result = client_data[client_id]
        n_samples = n_samples + result["n_samples"]
        mean = mean.add(result["mean"] * result["n_samples"], fill_value=0)

    norm_mean = mean / n_samples

    return norm_mean


def compute_global_std(client_data):
    logger.info("Compute global std")

    std = pd.Series(dtype="float64")
    n_samples = 0
    for client_id in client_data.keys():
        result = client_data[client_id]
        n_samples = n_samples + result["n_samples"]
        std = std.add(result["std"], fill_value=0)

    norm_std = np.sqrt(std / (n_samples - 1))

    return norm_std


def global_initialization(client_data):
    logger.info("Global initialization")

    D = []
    zr = pd.Series()
    n_samples = 0
    for client_id in client_data.keys():
        result = client_data[client_id]
        n_samples = n_samples + result["n_samples"]
        zlr = result["zlr"]
        distinct_times = result["distinct_times"]
        D.extend(distinct_times)
        zr = zr.add(zlr, fill_value=0)

    D = list(OrderedDict.fromkeys(D))
    D.sort()

    count_d = {}
    for t in D:
        t = str(float(t))
        val = 0
        for client_id in client_data.keys():
            result = client_data[client_id]
            n = result["numb_d_set"].get(t)
            if n is not None:
                val = val + int(n)
        count_d[t] = val

    return D, zr, count_d, n_samples


def newton_raphson_update(beta, gradient, hessian):
    """

    :param beta: beta calculated by the server
    :param gradient: first order derivative
    :param hessian: second order derivative
    :return: updated beta

    This method will apply the Newton-Raphson algorithm to update the new beta beta_(t) out of beta_(t-1)
    and its derivatives.
    """
    logger.info("Update beta (newton-fraphson)")

    inverse_l2 = np.linalg.inv(hessian)
    update_beta = beta - np.dot(inverse_l2, gradient)
    return update_beta


def calculate_convergence(beta, new_beta):
    """

    :param beta: beta_t
    :param new_beta: beta_t+1 calculated by newton raphson model
    :return: convergence (log-relative error)

    This method will calculate the the sum of the values of beta - the sum of the values of new_beta.
    """
    logger.info("Calculate convergence")

    convergence = beta.sum() - new_beta.sum()
    return convergence


def calculate_variance_matrix(hessian, norm_std, zr):
    """

    :param hessian: second order derivative
    :param norm_std: standard deviation
    :return: variance matrix
    """
    logger.info("Compute variance matrix")

    if hessian.size > 0:
        variance_matrix_ = pd.DataFrame(
            -np.linalg.inv(hessian) / np.outer(norm_std, norm_std), index=zr.axes[0],
            columns=zr.axes[0]
        )
    else:
        variance_matrix_ = pd.DataFrame(index=zr.axes[0], columns=zr.axes[0])
    return variance_matrix_


def calculate_standard_errors(variance_matrix_, params_):
    """

    :return: Pandas Series of the standard errors

    This method calculates the standard errors out of the variance matrix.
    """
    logger.info("Compute standard errors")

    se = np.sqrt(variance_matrix_.values.diagonal())
    return pd.Series(se, name="se", index=params_.index)


def calculate_z_values(params_, standard_errors_) -> pd.Series:
    """

    :return: z-values

    This method calculates the z-values.
    """
    logger.info("Compute z-values")

    return params_ / standard_errors_


def calculate_p_values(params_, standard_errors_) -> np.ndarray:
    """

    :return: pvalues

    This method calculates the pvalues.
    """
    logger.info("Compute p-values")

    U = calculate_z_values(params_, standard_errors_) ** 2
    return stats.chi2.sf(U, 1)


def calculate_confidence_intervals(alpha, standard_errors_, params_) -> pd.DataFrame:
    """

    :param alpha: normally 0.05
    :return: confidence intervals (the 95% lower and upper bound)
    """
    logger.info("Compute confidence intervals")

    ci = 100 * (1 - alpha)
    z = inv_normal_cdf(1 - alpha / 2)
    se = standard_errors_
    hazards = params_.values
    return pd.DataFrame(
        np.c_[hazards - z * se, hazards + z * se],
        columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
        index=params_.index,
    )


def summary(alpha, params_, hazard_ratios_, standard_errors_, confidence_intervals_, p_values_) -> pd.DataFrame:
    """

    :return: summary dataframe with several values

    This method will join all the calculated values to one dataframe as a summary.
    """
    logger.info("Compute summary")

    ci = 100 * (1 - alpha)
    z = inv_normal_cdf(1 - alpha / 2)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        df = pd.DataFrame(index=params_.index)
        df["coef"] = params_
        df["exp(coef)"] = hazard_ratios_
        df["se(coef)"] = standard_errors_
        df["coef lower %g%%" % ci] = confidence_intervals_["%g%% lower-bound" % ci]
        df["coef upper %g%%" % ci] = confidence_intervals_["%g%% upper-bound" % ci]
        df["exp(coef) lower %g%%" % ci] = hazard_ratios_ * np.exp(-z * standard_errors_)
        df["exp(coef) upper %g%%" % ci] = hazard_ratios_ * np.exp(z * standard_errors_)
        df["z"] = calculate_z_values(params_, standard_errors_)
        df["p"] = p_values_
        df["-log2(p)"] = -np.log2(df["p"])
        return df


def get_efron_values(client_data, zr, D, count_d, smpc=False):
    """
    :param global_stat: aggregated statistics of all clients
    Calculate the hessian and gradient out of the aggregated statistics from the slaves.
    """
    covariates = zr.axes[0]
    d = len(covariates)

    if not smpc:
        global_i1 = {}
        global_i2 = {}
        global_i3 = {}

        for client_id in client_data.keys():
            client = client_data[client_id]

            last_i1 = 0
            last_i2 = np.zeros((d,))
            last_i3 = np.zeros((d, d))

            # send sites an updated beta and sites calculate aggregated statistics
            i1 = client["is"][0]
            i2 = client["is"][1]
            i3 = client["is"][2]

            for time in sorted(D, reverse=True):
                np.set_printoptions(precision=30)

                t = str(float(time))
                if t in i1:
                    if t in global_i1:
                        df = global_i1[t]
                        global_i1[t] = df + i1[t]
                        i2_t = i2[t]
                        i2_t = np.array(i2_t)
                        df = global_i2[t]
                        global_i2[t] = df + i2_t
                        i3_t = i3[t]
                        i3_t = np.array(i3_t)
                        df = global_i3[t]
                        global_i3[t] = df + i3_t

                    else:
                        i2_t = i2[t]
                        i2_t = np.array(i2_t)
                        i3_t = i3[t]
                        i3_t = np.array(i3_t)
                        global_i1[t] = i1[t]
                        global_i2[t] = i2_t
                        global_i3[t] = i3_t

                    last_i1 = i1[t]
                    last_i2 = i2_t
                    last_i3 = i3_t
                # if time is not in i1 we have to add the value of the key time-1 of i1 to global_i1
                else:
                    if t in global_i1:
                        global_i1[t] = global_i1[t] + last_i1
                        global_i2[t] = global_i2[t] + last_i2
                        global_i3[t] = global_i3[t] + last_i3
                    else:
                        global_i1[t] = last_i1
                        global_i2[t] = last_i2
                        global_i3[t] = last_i3
    else:
        global_i1 = client_data["i1"]
        global_i2 = client_data["i2"]
        global_i3 = client_data["i3"]

    d1 = np.zeros((d,))
    d2 = np.zeros((d, d))

    for time in D:
        t = str(float(time))
        Dki = count_d[t]
        numer = global_i2[t]  # in global_i1 and global_i2 -> keys are strings
        denom = 1.0 / np.array([global_i1[t]])
        summand = numer * denom[:, None]
        d1 = d1 + Dki * summand.sum(0)
        a1 = global_i3[t] * denom
        a2 = np.dot(summand.T, summand)
        d2 = d2 + Dki * (a2 - a1)

    # first order derivative
    zr = zr.to_numpy()
    gradient = zr - d1
    hessian = d2

    return hessian, gradient


def iteration_update(client_data, beta, zr, converging, step_sizer, step_size, iteration, n, count_d, D,
                     precision=1e-07, penalization=0.0, l1_ratio=0.0, max_steps=500, smpc=False):
    """
    Update the model parameter beta and hessian.
    """
    soft_abs = lambda x, a: 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))
    penalizer = (lambda beta, a: n * 0.5 * penalization * (
                 l1_ratio * soft_abs(beta, a).sum() + (1 - l1_ratio) * (beta ** 2).sum()))
    d_penalizer = elementwise_grad(penalizer)
    dd_penalizer = elementwise_grad(d_penalizer)

    covariates = zr.axes[0]
    d = len(covariates)

    iteration = iteration + 1
    h, g = get_efron_values(client_data, zr, D, count_d, smpc)

    if penalization > 0:
        g -= d_penalizer(beta, 1.3 ** iteration)
        h[np.diag_indices(d)] -= dd_penalizer(beta, 1.3 ** iteration)

    try:
        inv_h_dot_g = spsolve(-h, g, assume_a="pos", check_finite=False)
    except (ValueError, LinAlgError) as e:
        if "infs or NaNs" in str(e):
            raise ConvergenceError(
                """Hessian or gradient contains nan or inf value(s). Convergence halted. {0}""".format(
                    CONVERGENCE_DOCS),
                e,
            )
        elif isinstance(e, LinAlgError):
            raise ConvergenceError(
                """Convergence halted due to matrix inversion problems. Suspicion is high collinearity. {0}""".format(
                    CONVERGENCE_DOCS
                ),
                e,
            )
        else:
            # something else?
            raise e

    delta = inv_h_dot_g
    hessian, _ = h, g
    if delta.size > 0:
        norm_delta = norm(delta)
    else:
        norm_delta = 0

    newton_decrement = g.dot(inv_h_dot_g) / 2

    # convergence criteria
    if norm_delta < precision:
        print("Norm delta smaller than precision")
        converging, success = False, True
    elif newton_decrement < precision:
        print("Newton decrement smaller than precision")
        converging, success = False, True
    elif iteration >= max_steps:
        # 50 iterations steps with N-R is a lot.
        # Expected convergence is ~10 steps
        converging, success = False, False
    elif step_size <= 0.00001:
        converging, success = False, False

        # report to the user problems that we detect.
    if not converging and success and norm_delta > 0.1:
        warnings.warn(
            "Newton-Rhaphson convergence completed successfully but norm(delta) is still high, %.3f. "
            "This may imply non-unique solutions to the maximum likelihood. "
            "Perhaps there is collinearity or complete separation in the dataset?\n"
            % norm_delta,
            exceptions.ConvergenceWarning,
        )
    elif not converging and not success:
        warnings.warn(
            "Newton-Rhaphson failed to converge sufficiently. {0}".format(CONVERGENCE_DOCS),
            exceptions.ConvergenceWarning
        )

    logger.info(
        f'Iteration {iteration}: norm_delta = {round(norm_delta, 5)}, step_size = {round(step_size, 4)}, '
        f'newton_decrement = {round(newton_decrement, 5)}')

    step_size = step_sizer.update(norm_delta).next()
    beta += step_size * delta

    return beta, converging, hessian, step_size, iteration, delta


def create_summary(norm_std, beta, zr, hessian, alpha=0.05):
    logger.info("Create summary")

    params_ = beta / norm_std.values
    params_ = pd.Series(params_, index=zr.axes[0], name="coef")

    # calculate other features for the summary
    hazard_ratios_ = pd.Series(np.exp(params_), index=zr.axes[0], name="exp(coef)")
    variance_matrix_ = calculate_variance_matrix(hessian, norm_std, zr)
    standard_errors_ = calculate_standard_errors(variance_matrix_, params_)
    p_values_ = pd.Series(calculate_p_values(params_, standard_errors_), index=zr.axes[0])
    confidence_intervals_ = calculate_confidence_intervals(alpha, standard_errors_, params_)

    # combine the values to one big dataframe as a summary
    df_summary = summary(alpha, params_, hazard_ratios_, standard_errors_, confidence_intervals_,
                         p_values_)
    return df_summary, params_, standard_errors_


def calculate_concordance_index(client_data):
    """
    :return: concordance index

    This method calculates the concordance index of the model.
    It is a measure of the predictive accuracy of the fitted model onto the training dataset.
    """
    logger.info("Compute c-index")

    global_c_index = 0.0
    sample_number = 0
    for client_id in client_data.keys():
        data = client_data[client_id]
        sample_number = sample_number + data["sample_number"]
        if data["c-index"] is not None:
            global_c_index += data["c-index"] * data["sample_number"]

    try:
        global_c_index = global_c_index / sample_number
    except Exception:
        global_c_index = None

    return global_c_index, sample_number
