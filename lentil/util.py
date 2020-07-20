import sys
from collections import defaultdict

import pandas as pd

from lentil import datatools
from lentil import est
from lentil import models


def interaction_history_from_assistments_data_set(
        data,
        duration_column='timestep',
        module_id_column='problem_id'):
    """
    Parse dataframe of assistments interactions into an interaction history

    :param pd.DataFrame assistments_data: A raw history from assistments
    :param str duration_column: Column to use as interaction duration
    :param str module_id_column: Column to use as module_id
    :rtype: datatools.InteractionHistory
    :return: An interaction history
    """
    # sort by order_id
    data.sort_values(by='order_id', inplace=True, axis=0)

    # get relevant columns and rename them
    data = data[['user_id', 'correct', duration_column, module_id_column]]
    data.columns = ['user_id', 'outcome', 'duration', 'module_id']

    # only keep interactions with binary outcomes and positive response times
    data = data[((data['outcome'] == 1) | (data['outcome'] == 0)) & (data['duration'] > 0)]

    # cast outcomes from int to bool
    data['outcome'] = data['outcome'].apply(lambda x: x == 1)

    # map response times from milliseconds to seconds
    data['duration'] = data['duration'].apply(lambda x: x / 1000)

    # existing interactions are all assessment interactions
    data['module_type'] = [datatools.AssessmentInteraction.MODULETYPE] * len(data)

    # add timesteps
    timesteps = [None] * len(data)
    student_timesteps = defaultdict(int)
    for i, (_, ixn) in enumerate(data.iterrows()):
        student_timesteps[ixn['user_id']] += 1
        timesteps[i] = student_timesteps[ixn['user_id']]
    data['timestep'] = timesteps

    # add artificial lesson interactions
    lesson_data = data.copy(deep=True)
    lesson_data['module_type'] = [datatools.LessonInteraction.MODULETYPE] * len(data)

    return datatools.InteractionHistory(
        pd.concat([data, lesson_data]),
        sort_by_timestep=True)


def filter_history(history, min_num_ixns=5, max_num_ixns=sys.maxint):
    """
    Filter history for students with histories of bounded length,
    and modules with enough interactions

    :param datatools.InteractionHistory history: An interaction history
    :param int min_num_ixns: Minimum number of timesteps in student history,
        and minimum number of interactions for module

    :param int max_num_ixns: Maximum number of timesteps in student history
    :rtype: datatools.InteractionHistory
    :return: A filtered interaction history
    """
    students = set(history.data['user_id'][(
                                                   history.data['timestep'] > min_num_ixns) & (
                                                   history.data[
                                                       'module_type'] == datatools.AssessmentInteraction.MODULETYPE)])
    students -= set(history.data['user_id'][history.data['timestep'] >= max_num_ixns])

    modules = {module_id for module_id, group in history.data.groupby('module_id') if len(group) > min_num_ixns}

    return datatools.InteractionHistory(
        history.data[(history.data['user_id'].isin(students)) & (
            history.data['module_id'].isin(modules))],
        reindex_timesteps=True)


def build_1pl_irt_model(history, filtered_history, split_history=None):
    model = models.OneParameterLogisticModel(
        filtered_history, select_regularization_constant=True)
    model.fit()
    return model


def build_2pl_irt_model(history, filtered_history, split_history=None):
    model = models.TwoParameterLogisticModel(
        filtered_history, select_regularization_constant=True)
    model.fit()
    return model


def meta_build_mirt_model(regularization_constant=1e-3, dims=2):
    def build_mirt_model(history, filtered_history, split_history=None):
        model = models.MIRTModel(history, dims)
        estimator = est.MIRTMAPEstimator(
            regularization_constant=regularization_constant,
            ftol=1e-4,
            verify_gradient=False,
            debug_mode_on=True,
            filtered_history=filtered_history,
            split_history=split_history)
        model.fit(estimator)
        return model

    return build_mirt_model


def build_student_biased_coin_model(history, filtered_history, split_history=None):
    model = models.StudentBiasedCoinModel(history, filtered_history)
    model.fit()
    return model


def build_assessment_biased_coin_model(history, filtered_history, split_history=None):
    model = models.AssessmentBiasedCoinModel(history, filtered_history)
    model.fit()
    return model


def build_embedding(
        embedding_kwargs,
        estimator,
        history,
        filtered_history,
        split_history=None):
    model = models.EmbeddingModel(history, **embedding_kwargs)

    estimator.filtered_history = filtered_history
    if split_history is not None:
        estimator.split_history = split_history

    model.fit(estimator)

    return model


def meta_build_embedding(
        d=2,
        using_lessons=True,
        using_prereqs=True,
        using_bias=True,
        regularization_constant=1e-3,
        using_scipy=True,
        learning_update_variance_constant=0.5,
        forgetting_penalty_term_constant=0.,
        tv_luv_model=None,
        forgetting_model=None,
        using_graph_prior=None,
        graph=None,
        graph_regularization_constant=None):
    embedding_kwargs = {
        'embedding_dimension': d,
        'using_lessons': using_lessons,
        'using_prereqs': using_prereqs,
        'using_bias': using_bias,
        'learning_update_variance_constant': learning_update_variance_constant,
        'tv_luv_model': tv_luv_model,
        'forgetting_model': forgetting_model,
        'forgetting_penalty_term_constant': forgetting_penalty_term_constant,
        'using_graph_prior': using_graph_prior,
        'graph': graph,
        'graph_regularization_constant': graph_regularization_constant
    }

    gradient_descent_kwargs = {
        'using_adagrad': False,
        'eta': 0.001,
        'eps': 0.1,
        'rate': 0.005,
        'verify_gradient': False,
        'ftol': 1e-3,
        'max_iter': 1000,
        'num_checkpoints': 100
    }

    estimator = est.EmbeddingMAPEstimator(
        regularization_constant=regularization_constant,
        using_scipy=using_scipy,
        gradient_descent_kwargs=gradient_descent_kwargs,
        verify_gradient=False,
        debug_mode_on=True,
        ftol=1e-4)

    return (lambda *args, **kwargs: build_embedding(
        embedding_kwargs,
        estimator,
        *args,
        **kwargs))



