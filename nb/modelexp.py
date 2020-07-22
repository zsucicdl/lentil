from __future__ import division

import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from lentil import datatools
from lentil import est
from lentil import evaluate
from lentil import models

sns.set_style('whitegrid')

import logging
from lentil.util import *

logging.getLogger().setLevel(logging.DEBUG)

history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_data.csv'
df = pd.read_csv(history_path,
                 dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int, 'problem_id': int,
                        'original': int, 'correct': int, 'attempt_count': int, 'ms_first_response': int,
                        'tutor_mode': 'string', 'answer_type': 'string', 'sequence_id': int, 'student_class_id': int,
                        'position': int, 'type': 'string', 'base_sequence_id': int, 'skill_id': float,
                        'skill_name': 'string',
                        'teacher_id': int, 'school_id': int, 'hint_count': int, 'hint_total': int, 'overlap_time': int,
                        'template_id': int, 'answer_id': int, 'answer_text': 'string', 'first_action': int,
                        'bottom_hint': int, 'opportunity': int, 'opportunity_original': int
                        },
                 usecols=['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 'original', 'correct',
                          'attempt_count', 'ms_first_response', 'tutor_mode', 'answer_type', 'sequence_id',
                          'student_class_id', 'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name',
                          'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time', 'template_id',
                          'first_action', 'opportunity', ])
print("Input done.")

unfiltered_history = interaction_history_from_assistments_data_set(
    df,
    module_id_column='problem_id',
    duration_column='ms_first_response')

# apply the filter a couple of times, since removing student histories
# may cause certain modules to drop below the min_num_ixns threshold,
# and removing modules may cause student histories to drop below
# the min_num_ixns threshold
REPEATED_FILTER = 3  # number of times to repeat filtering
history = reduce(
    lambda acc, _: filter_history(acc, min_num_ixns=75, max_num_ixns=1000),
    range(REPEATED_FILTER), unfiltered_history)

df = history.data

embedding_dimension = 2

model = models.EmbeddingModel(
    history,
    embedding_dimension,
    using_prereqs=True,
    using_lessons=True,
    using_bias=True,
    learning_update_variance_constant=0.5)

estimator = est.EmbeddingMAPEstimator(
    regularization_constant=1e-3,
    using_scipy=True,
    verify_gradient=False,
    debug_mode_on=True,
    ftol=1e-3)

model.fit(estimator)

print "Training AUC = %f" % (evaluate.training_auc(
    model, history, plot_roc_curve=True))

split_history = history.split_interactions_by_type()
timestep_of_last_interaction = split_history.timestep_of_last_interaction

NUM_STUDENTS_TO_SAMPLE = 10
for user_id in random.sample(df['user_id'].unique(), NUM_STUDENTS_TO_SAMPLE):
    user_idx = history.idx_of_user_id(user_id)

    timesteps = range(1, timestep_of_last_interaction[user_id] + 1)

    for i in range(model.embedding_dimension):
        plt.plot(timesteps, model.student_embeddings[user_idx, i, timesteps],
                 label='Skill %d' % (i + 1))

    norms = np.linalg.norm(model.student_embeddings[user_idx, :, timesteps], axis=1)
    plt.plot(timesteps, norms, label='norm')

    plt.title('user_id = %s' % user_id)
    plt.xlabel('Timestep')
    plt.ylabel('Skill')
    plt.legend(loc='upper right')
    plt.show()

assessment_norms = np.linalg.norm(model.assessment_embeddings, axis=1)

plt.xlabel('Assessment embedding norm')
plt.ylabel('Frequency (number of assessments)')
plt.hist(assessment_norms, bins=20)
plt.show()


def get_pass_rates(grouped):
    """
    Get pass rate for each group

    :param pd.GroupBy grouped: A grouped dataframe
    :rtype: dict[str, float]
    :return: A dictionary mapping group name to pass rate
    """
    pass_rates = {}
    for name, group in grouped:
        vc = group['outcome'].value_counts()
        if True not in vc:
            pass_rates[name] = 0
        else:
            pass_rates[name] = vc[True] / len(group)
    return pass_rates


grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('module_id')
pass_rates = get_pass_rates(grouped)

assessment_norms = [np.linalg.norm(model.assessment_embeddings[history.idx_of_assessment_id(assessment_id), :]) for
                    assessment_id in pass_rates]

plt.xlabel('Assessment pass rate')
plt.ylabel('Assessment embedding norm')
plt.scatter(pass_rates.values(), assessment_norms)
plt.show()

grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('module_id')
pass_rates = get_pass_rates(grouped)

bias_minus_norm = [model.assessment_biases[history.idx_of_assessment_id(
    assessment_id)] - np.linalg.norm(
    model.assessment_embeddings[history.idx_of_assessment_id(
        assessment_id), :]) for assessment_id in pass_rates]

plt.xlabel('Assessment pass rate')
plt.ylabel('Assessment bias - Assessment embedding norm')
plt.scatter(pass_rates.values(), bias_minus_norm)
plt.show()

grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('user_id')
pass_rates = get_pass_rates(grouped)

biases = [model.student_biases[history.idx_of_user_id(
    user_id)] for user_id in pass_rates]

plt.xlabel('Student pass rate')
plt.ylabel('Student bias')
plt.scatter(pass_rates.values(), biases)
plt.show()

lesson_norms = np.linalg.norm(model.lesson_embeddings, axis=1)

plt.xlabel('Lesson embedding norm')
plt.ylabel('Frequency (number of lessons)')
plt.hist(lesson_norms, bins=20)
plt.show()

prereq_norms = np.linalg.norm(model.prereq_embeddings, axis=1)

plt.xlabel('Prereq embedding norm')
plt.ylabel('Frequency (number of lessons)')
plt.hist(prereq_norms, bins=20)
plt.show()

plt.xlabel('Lesson embedding norm')
plt.ylabel('Prereq embedding norm')
plt.scatter(prereq_norms, lesson_norms)
plt.show()

timesteps = range(model.student_embeddings.shape[2])
avg_student_norms = np.array(np.linalg.norm(np.mean(model.student_embeddings, axis=0), axis=0))

plt.xlabel('Timestep')
plt.ylabel('Average student embedding norm')
plt.plot(timesteps, avg_student_norms)
plt.show()
