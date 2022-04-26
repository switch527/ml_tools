import numpy as np


def get_results(tune_search):
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
