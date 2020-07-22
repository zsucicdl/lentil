from __future__ import division

import pickle
import os

from matplotlib import pyplot as plt
from scipy import linalg
import numpy as np
import networkx as nx

import seaborn as sns

sns.set_style('whitegrid')

from lentil import datatools

import logging
import pandas as pd
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



idx_of_module_id = {k: i for i, k in enumerate(df['module_id'].unique())}
num_modules = len(idx_of_module_id)
print "Number of unique modules = %d" % num_modules

# compute adjacency matrix of flow graph

# sometimes a student history contains a module id
# multiple times (for assessment and lesson interactions)
IGNORE_REPEATED_MODULE_IDS = True

X = np.zeros((num_modules, num_modules))
grouped = df.groupby('user_id')['module_id']
for user_id, group in grouped:
    module_idxes = group.map(idx_of_module_id).values

    if IGNORE_REPEATED_MODULE_IDS:
        filtered_module_idxes = []
        module_idxes_seen = set()
        for module_idx in module_idxes:
            if module_idx in module_idxes_seen:
                continue
            filtered_module_idxes.append(module_idx)
            module_idxes_seen |= {module_idx}

    # okay because module transitions are never repeated in this dataset
    # if that's not true, then use np.add.at
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html
    X[module_idxes[:-1], module_idxes[1:]] += 1

# is the Markov chain ergodic?
# i.e., is the flow graph strongly connected?
G = nx.from_numpy_matrix(X, create_using=nx.DiGraph())

sc = nx.strongly_connected_components(G)

print "Sizes of strongly connected components:"
print [len(x) for x in sc]

# compute transition probability matrix of Markov chain
P = X / X.sum(axis=1)[:, np.newaxis]

# estimate stationary distribution of Markov chain
stationary_distrn = np.diag(np.linalg.matrix_power(P, 2 ** 15))

prev = P
N = 15
diffs = [None] * N
for i in range(N):
    nP = np.dot(prev, prev)
    diffs[i] = np.linalg.norm(np.diag(nP) - np.diag(prev))
    prev = nP

plt.xlabel('n')
plt.ylabel('||diag(P^n)-diag(P^(n-1))||')
plt.plot(2 ** np.arange(0, N, 1), diffs, '-s')
plt.yscale('log')
plt.xscale('log')
plt.show()

entropy = -np.dot(stationary_distrn, np.nansum(P * np.log(P), axis=1))
print "Entropy = %f" % entropy
entropies=[]
entropies.append(entropy)
#
# output_path = os.path.join('results', 'entropy', 'grockit_entropy.pkl')
#
# with open(output_path, 'wb') as f:
#     pickle.dump(entropy, f, pickle.HIGHEST_PROTOCOL)
#
# data_sets = ['assistments_2009_2010', 'algebra_2006_2007',
#              'algebra_2005_2006', 'bridge_to_algebra_2006_2007', 'grockit']
#
# entropy_file_of_data_set = {k: os.path.join(
#     'results', 'entropy', '%s_entropy.pkl' % k) for k in data_sets}
#
# results_file_of_data_set = {k: os.path.join(
#     'results', 'last', '%s_results_lesion.pkl' % k) for k in data_sets}

entropies_of_models, results_of_models = [], []
entropies_of_models.append(entropy)
results_of_models.append(entropy)
# for ds in data_sets:
#     with open(entropy_file_of_data_set[ds], 'rb') as f:
#         entropies_of_models.append(pickle.load(f))
#     with open(results_file_of_data_set[ds], 'rb') as f:
#         results_of_models.append(pickle.load(f))

data_sets=['skill_builder_data.csv']

def make_plot(eps=1e-2):
    gains_of_models = [compute_gain_from_prereq_model(results) for results in results_of_models]

    plt.xlabel('Entropy of student paths')
    plt.ylabel(name_of_gain_metric)
    plt.scatter(entropies_of_models, gains_of_models)
    for e, g, ds in zip(entropies_of_models, gains_of_models, data_sets):
        plt.annotate(ds, (e + eps, g + eps))
    plt.show()


name_of_gain_metric = 'Relative AUC gain from prereq model'


def compute_gain_from_prereq_model(res):
    a = res.validation_auc_mean('d=2, without prereqs and bias')
    b = res.validation_auc_mean('d=2, without prereqs, with bias')
    c = res.validation_auc_mean('d=2, with prereqs, without bias')
    d = res.validation_auc_mean('d=2, with prereqs and bias')
    return np.mean([(c - a) / a, (d - b) / b])


make_plot()

name_of_gain_metric = 'Relative AUC gain from prereq model (without bias)'


def compute_gain_from_prereq_model(res):
    a = res.validation_auc_mean('d=2, without prereqs and bias')
    c = res.validation_auc_mean('d=2, with prereqs, without bias')
    return (c - a) / a


make_plot()

name_of_gain_metric = 'Relative AUC gain from prereq model (with bias)'


def compute_gain_from_prereq_model(res):
    b = res.validation_auc_mean('d=2, without prereqs, with bias')
    d = res.validation_auc_mean('d=2, with prereqs and bias')
    return (d - b) / b


make_plot(eps=1e-3)

name_of_gain_metric = 'AUC gain from prereq model'


def compute_gain_from_prereq_model(res):
    a = res.validation_auc_mean('d=2, without prereqs and bias')
    b = res.validation_auc_mean('d=2, without prereqs, with bias')
    c = res.validation_auc_mean('d=2, with prereqs, without bias')
    d = res.validation_auc_mean('d=2, with prereqs and bias')
    return np.mean([c - a, d - b])


make_plot()
