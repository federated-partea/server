import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compute_dp_matrix(eps, d_n_matrix):
    logger.debug("Compute DP matrix")
    # Create partial matrix M
    m = pd.DataFrame(index=d_n_matrix.index, columns=['di', 'ri'])
    # t_1 = d_1 and n_1 known
    m["di"] = d_n_matrix["di"]
    m.iloc[0, 1] = d_n_matrix.iloc[0, 1]
    # Add Laplacian Noise
    z = m.copy()
    z_di = np.random.laplace(loc=0, scale=2 / eps, size=d_n_matrix.shape[0])
    z_ri = np.random.laplace(loc=0, scale=2 / eps, size=d_n_matrix.shape[0])
    z['di'] = z_di
    z['ri'] = z_ri
    m_ = pd.DataFrame(index=d_n_matrix.index, columns=['di', 'ri'])
    m_['di'] = m['di'].astype(float) + z['di'].astype(float)
    m_['ri'] = m['ri'] + z['ri'].astype(float)

    # compute differentially number at risk
    r_last = m_.iloc[0, 1]
    d_last = m_.iloc[0, 0]
    for j in range(1, d_n_matrix.shape[0]):
        r = r_last - d_last
        d = m_.iloc[j, 0]
        if r < 0:
            m_.iloc[j, 1] = 0
        else:
            m_.iloc[j, 1] = r
        if d < 0:
            m_.iloc[j, 0] = 0
        r_last = r
        d_last = d
    # Replace negative values by 0 for di and end df if ri is 0
    m_ = m_.reset_index()
    ri_zero_index = (m_["ri"] == 0).idxmax()
    if ri_zero_index != 0:
        m_ = m_.iloc[0:ri_zero_index - 1, :]
    m_ = m_.rename({"ri": "ni"}, axis=1)
    return m_.clip(lower=0)


def compute_survival_function(m, surv_label="KM_estimate"):
    logger.debug("Compute Survival function")
    m = m[m['ni'] != 0]
    m["1-(di/ni)"] = (1 - m["di"] / m["ni"])
    m.dropna(inplace=True)
    m.reset_index(inplace=True)
    m.rename({"index": "t"}, axis=1, inplace=True)
    si = []
    for i in m.index:
        si.append(np.product(m.loc[0:i, "1-(di/ni)"].to_list()))
    m["si"] = si
    if m.iloc[0, 0] != 0.0:
        m.index = m.index + 1
        m.loc[0, :] = [0.0, 0.0, m.shape[0], 1.0, 1.0]
        m.sort_index(inplace=True)
    m.rename({"t": "timeline", "si": surv_label}, axis=1, inplace=1)
    m.set_index("timeline", inplace=True)

    return m[surv_label]


def compute_hazard_function(m, cum_label="NA_estimate"):
    logger.debug("Compute hazard function")
    m = m[m['ni'] != 0]
    m["di/ni"] = (m["di"] / m["ni"])
    m.dropna(inplace=True)
    m.reset_index(inplace=True)
    m.rename({"index": "t"}, axis=1, inplace=True)
    hi = []
    for t in m.index:
        hi.append(np.sum(m.loc[0:t, "di/ni"].to_list()))
    m["hi"] = hi
    if m.iloc[0, 0] != 0.0:
        m.index = m.index + 1
        m.loc[0, :] = [0.0, 0.0, m.shape[0], 0.0, 0.0]
        m.sort_index(inplace=True)
    m.rename({"t": "timeline", "hi": cum_label}, axis=1, inplace=1)
    m.set_index("timeline", inplace=True)

    return m[cum_label]


def univariate_analysis(client_data, privacy_level, smpc: bool = False):
    logger.debug('Perform univariate analysis')
    time = {}
    data = {}
    n_samples = 0
    results = {"KM": None, "NA": None}
    if not smpc:
        for client_id in client_data.keys():
            client_result = client_data[client_id]['local_results']
            n_samples += client_data[client_id]['sample_number']
            for category in client_result.keys():
                if category not in data.keys():
                    data[category] = []
                    time[category] = []
                df = pd.DataFrame.from_dict(client_result[category])
                data[category].append(df)
                time[category] += list(df.index)
    else:
        for category in client_data.keys():
            if category not in data.keys():
                data[category] = []
                time[category] = []
            data[category].append(client_data[category])
            time[category] += list(client_data[category].index.astype(float))

    timelines = dict((k, sorted(set(v))) for k, v in time.items())
    aggregation_dfs = {}
    for category in data.keys():
        aggregation_df = pd.DataFrame(index=timelines[category], columns=["di", "ni"])
        for t in timelines[category]:
            di = 0
            ni = 0
            for c in data[category]:
                try:
                    di = di + float(c.loc[t, "di"])
                except KeyError:
                    pass
                try:
                    ni = ni + float(c.loc[t, "ni"])
                except KeyError:
                    indices = list(c.index)
                    try:
                        res = list(map(lambda i: i > t, indices)).index(True)
                        ni = ni + float(c.loc[indices[res], "ni"])
                    except ValueError:
                        pass

                aggregation_df.loc[t, "di"] = di
                aggregation_df.loc[t, "ni"] = ni

        aggregation_df = aggregation_df.dropna()
        if privacy_level != 0:
            eps = privacy_level
            aggregation_df = compute_dp_matrix(eps, aggregation_df)
        aggregation_dfs[category] = aggregation_df
        surv_label = str(category)
        cum_label = str(category)
        survival_function = compute_survival_function(aggregation_df.copy(), surv_label)
        cum_hazard_rate = compute_hazard_function(aggregation_df.copy(), cum_label)

        if results["KM"] is None:
            results["KM"] = survival_function.to_frame()
        else:
            results["KM"] = pd.concat([results["KM"], survival_function], axis=1, join='outer')
        if results["NA"] is None:
            results["NA"] = cum_hazard_rate.to_frame()
        else:
            results["NA"] = pd.concat([results["NA"], cum_hazard_rate], axis=1, join='outer')
    if not smpc:
        return results, aggregation_dfs, n_samples
    else:
        return results, aggregation_dfs


def logrank_test(df1, df2):
    logger.debug("Perform logrank test")

    logrank_df = pd.merge(left=df1, right=df2, left_index=True, right_index=True,
                          how='outer', suffixes=('_0', '_1'))
    logrank_df.iloc[:, [0, 2]] = logrank_df.iloc[:, [0, 2]].fillna(0)

    logrank_df = logrank_df.fillna(method='bfill')
    logrank_df = logrank_df.fillna(0)
    logrank_df = logrank_df.div(2)

    # compute the factors needed (from lifelines)
    N_j = logrank_df[["di_0", "di_1"]].sum(0).values
    n_ij = logrank_df[["ni_0", "ni_1"]]
    d_i = logrank_df["di_0"] + logrank_df["di_1"]
    n_i = logrank_df["ni_0"] + logrank_df["ni_1"]
    ev = n_ij.mul(d_i / n_i, axis="index").sum(0)

    # vector of observed minus expected
    Z_j = N_j - ev

    assert abs(Z_j.sum()) < 10e-8, "Sum is not zero."  # this should move to a test eventually.

    # compute covariance matrix
    factor = (((n_i - d_i) / (n_i - 1)).replace([np.inf, np.nan], 1)) * d_i / n_i ** 2
    n_ij["_"] = n_i.values
    V_ = n_ij.mul(np.sqrt(factor), axis="index").fillna(0)
    V = -np.dot(V_.T, V_)
    ix = np.arange(2)
    V[ix, ix] = V[ix, ix] - V[-1, ix]
    V = V[:-1, :-1]

    # take the first n-1 groups
    U = Z_j.iloc[:-1] @ np.linalg.pinv(V[:-1, :-1]) @ Z_j.iloc[:-1]  # Z.T*inv(V)*Z
    # compute the p-values and tests
    return stats.chi2.sf(U, 1), U


def pairwise_logrank_test(data, alpha=0.05, method='bonferroni'):
    logger.debug("Perform pairwise logrank test")

    my_index = pd.MultiIndex(levels=[[], []],
                             codes=[[], []],
                             names=[u'Category 1', u'Category 2'])
    my_columns = [u'test_statistic', u'p', u'log(p)', u'p_bonf', u'log(p_bonf)']
    p_matrix = pd.DataFrame(index=my_index, columns=my_columns)
    for category, category2 in itertools.combinations(list(data.keys()), 2):
        p, U = logrank_test(data[category2], data[category])
        p_matrix.loc[(category2, category), :] = [U, p, np.nan, np.nan, np.nan]
    p_corrected = multipletests(p_matrix.loc[:, "p"].to_list(), alpha=alpha, method=method, is_sorted=False,
                                returnsorted=False)
    p_matrix["p_bonf"] = p_corrected[1]
    p_matrix["log(p)"] = p_matrix.loc[:, "p"].apply(lambda x: -np.log2(x))
    p_matrix["log(p_bonf)"] = p_matrix.loc[:, "p"].apply(lambda x: -np.log2(x))
    p_matrix = p_matrix.astype("float64")
    p_matrix = p_matrix.reset_index()
    if "int" in str(p_matrix.dtypes["Category 1"]):
        p_matrix = p_matrix.sort_values(['Category 1', 'Category 2'], ascending=[1, 1])
    p_matrix = p_matrix.set_index(['Category 1', 'Category 2'])

    return p_matrix
