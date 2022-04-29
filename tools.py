import numpy as np
from pandas import DataFrame
from pandas import concat


def get_results(tune_search):
    """
    :param tune_search:
    :return:
    """
    params = tune_search.best_params_
    cv_results = tune_search.cv_results_
    refit_metric = tune_search.refit
    best_index = tune_search.best_index
    metrics = list(tune_search.scoring.keys())

    results = dict()
    results['params'] = params
    results['refit_metric'] = refit_metric
    results['best_refit_index'] = best_index

    for i in metrics:
        if i == refit_metric:
            results[i + '_refit'] = cv_results['mean_test_' + i][best_index]
        else:
            if cv_results['mean_test_' + i][best_index] == max(cv_results['mean_test_' + i]):
                results[i + '_refit_best'] = cv_results['mean_test_' + i][best_index]
            else:
                results[i + '_refit'] = cv_results['mean_test_' + i][best_index]
                results[i + '_best'] = [max(cv_results['mean_test_' + i]),
                                        cv_results['mean_test_' + i][best_index] - max(cv_results['mean_test_' + i]),
                                        int(np.where(
                                            cv_results['mean_test_' + i] == max(cv_results['mean_test_' + i]))[
                                                0])]
    return results


def series_to_supervised(data, n_in=2, n_out=1, dropnan=True, ascending=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    data.sort_index(ascending=True, inplace=True)

    columns = list(data.columns)

    cols, names = list(), list()
    cols.append(data)
    # input sequence (t-n, ... t-1)
    for i in range(1, n_in, 1):
        column_names = dict()

        for ii in columns:
            column_names[ii] = str(ii) + '(t-' + str(i) + ')'

        cols.append(data.shift(i).rename(columns=column_names))

    agg = pd.concat(cols, axis=1)

    if dropnan:
        agg.dropna(inplace=True)

    if not ascending:
        agg.sort_index(ascending=False, inplace=True)

    return agg


def series_to_supervised_copy(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg