def get_results(TuneSearch):
    params = TuneSearch.best_params_
    cv_results = TuneSearch.cv_results_
    refit_metric = TuneSearch.refit
    best_index = TuneSearch.best_index
    metrics = list(TuneSearch.scoring.keys())

    results = {}
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
                                            cv_results['mean_test_' + i] == max(search.cv_results_['mean_test_' + i]))[
                                                0])]

    return results