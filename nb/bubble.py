from __future__ import division

import logging
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

from lentil import datatools
from lentil import est
from lentil import models
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

unfiltered_history = interaction_history_from_assistments_data_set(
    df,
    module_id_column='problem_id',
    duration_column='ms_first_response')

REPEATED_FILTER = 3  # number of times to repeat filtering
history = reduce(
    lambda acc, _: filter_history(acc, min_num_ixns=75, max_num_ixns=1000),
    range(REPEATED_FILTER), unfiltered_history)

# history = datatools.InteractionHistory(filtered_history)
df = history.data
duration = history.duration()

num_students = history.num_students()
num_left_out_students = int(0.3 * num_students)
left_out_user_ids = {history.id_of_user_idx(
    user_idx) for user_idx in random.sample(
    range(num_students), num_left_out_students)}

left_out_ixns = df['user_id'].isin(left_out_user_ids)
left_in_ixns = ~left_out_ixns
left_in_modules = set(df[left_in_ixns]['module_id'].unique())

print "Number of unique students = %d" % (len(df['user_id'].unique()))
print "Number of unique modules = %d" % (len(df['module_id'].unique()))

# these constraints speed up bubble collection
MIN_BUBBLE_LENGTH = 2  # number of interactions
MAX_BUBBLE_LENGTH = 20  # number of interactions

grouped = history.data.groupby('user_id')

# dict[(str, str), dict[tuple, int]]
# (start lesson id, final assessment id) -> 
# dict[(lesson id, lesson id, ...) -> 
# number of students who took this path]
bubble_paths = {}

# dict[(str, str, (str, str, ...)), list[str]]
# (start lesson id, final assessment id, lesson sequence) -> 
# [ids of students who took this path]
bubble_students = defaultdict(list)

# dict[(str, str, (str, str, ...)), set[str]]
# (start lesson id, final assessment id, lesson sequence) -> 
# [outcomes for students who took this path]
bubble_outcomes = defaultdict(list)

for user_id in left_out_user_ids:
    group = grouped.get_group(user_id)
    module_ids = list(group['module_id'])
    module_types = list(group['module_type'])
    outcomes = list(group['outcome'])
    for i, start_lesson_id in enumerate(module_ids):
        if module_types[i] != datatools.LessonInteraction.MODULETYPE:
            continue
        if start_lesson_id not in left_in_modules:
            continue
        for j, (final_assessment_id, module_type, outcome) in enumerate(
                zip(module_ids[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)],
                    module_types[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)],
                    outcomes[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)])):
            if final_assessment_id not in left_in_modules:
                break
            if module_type == datatools.AssessmentInteraction.MODULETYPE:
                lesson_seq = [x for x, y in zip(module_ids[i:(i + MIN_BUBBLE_LENGTH + j)],
                                                module_types[i:(i + MIN_BUBBLE_LENGTH + j)]) if
                              y == datatools.LessonInteraction.MODULETYPE]
                path = tuple(lesson_seq)
                if any(m not in left_in_modules for m in path):
                    break

                try:
                    bubble_paths[(start_lesson_id, final_assessment_id)][path] += 1
                except KeyError:
                    bubble_paths[(start_lesson_id, final_assessment_id)] = defaultdict(int)
                    bubble_paths[(start_lesson_id, final_assessment_id)][path] += 1

                bubble_students[(start_lesson_id, final_assessment_id, path)].append(user_id)
                bubble_outcomes[(start_lesson_id, final_assessment_id, path)].append(1 if outcome else 0)

MIN_NUM_STUDENTS_ON_PATH = 10

# dict[(str, str, (str, str, ...), (str, str, ...)]
# (start lesson id, final assessment id, path, other_path) -> 
# ([ids of students who took path], [ids of students who took other_path])
my_bubble_students = {}

# dict[(str, str, (str, str, ...), (str, str, ...)]
# (start lesson id, final assessment id, path, other_path) -> 
# ([outcomes for students who took path], [outcomes for students who took other_path])
my_bubble_outcomes = {}

for (start_lesson_id, final_assessment_id), d in bubble_paths.iteritems():
    paths = [path for path, num_students_on_path in d.iteritems() if num_students_on_path >= MIN_NUM_STUDENTS_ON_PATH]
    for i, path in enumerate(paths):
        for other_path in paths[(i + 1):]:
            if len(path) != len(other_path):
                # paths must have the same number of lesson interactions
                # in order to be part of a bubble
                continue
            my_bubble_students[(start_lesson_id, final_assessment_id, path, other_path)] = (
                bubble_students[(start_lesson_id, final_assessment_id, path)],
                bubble_students[(start_lesson_id, final_assessment_id, other_path)])
            my_bubble_outcomes[(start_lesson_id, final_assessment_id, path, other_path)] = (
                bubble_outcomes[(start_lesson_id, final_assessment_id, path)],
                bubble_outcomes[(start_lesson_id, final_assessment_id, other_path)])

bubble_paths = None  # clear memory

MIN_NUM_STUDENTS_IN_BUBBLE = 10  # minimum num students on a branch
MIN_AVG_BUBBLE_PATH_LENGTH = 2  # minimum branch length


def is_valid_bubble(k):
    students_on_path, students_on_other_path = my_bubble_students[k]
    pass_rate = np.mean(sum(my_bubble_outcomes[k], []))
    num_students_in_bubble = min(len(students_on_path), len(students_on_other_path))
    path_length = min(len(path), len(other_path))
    return pass_rate != 0 and pass_rate != 1 and num_students_in_bubble >= MIN_NUM_STUDENTS_IN_BUBBLE and path_length >= MIN_AVG_BUBBLE_PATH_LENGTH


is_valid_bubble_memo = {k: is_valid_bubble(k) for k in my_bubble_students}
print "Number of valid bubbles = %d" % sum(1 for v in is_valid_bubble_memo.itervalues() if v)

# filter out invalid bubbles
ks = set()
for (mid, later_mid, path, other_path), is_valid in is_valid_bubble_memo.iteritems():
    if is_valid:
        ks |= {(mid, later_mid, path), (mid, later_mid, other_path)}

for k, is_valid in is_valid_bubble_memo.iteritems():
    mid, later_mid, path, other_path = k
    if not is_valid:
        my_bubble_students.pop(k, None)
        my_bubble_outcomes.pop(k, None)
        if (mid, later_mid, path) not in ks:
            bubble_students.pop((mid, later_mid, path), None)
            bubble_outcomes.pop((mid, later_mid, path), None)
        if (mid, later_mid, other_path) not in ks:
            bubble_students.pop((mid, later_mid, other_path), None)
            bubble_outcomes.pop((mid, later_mid, other_path), None)

plt.xlabel('Size of bubble experiment (number of students in smaller arm)')
plt.ylabel('Frequency (number of bubbles)')
plt.hist([min(len(v[0]), len(v[1])) for k, v in my_bubble_students.iteritems()])
plt.show()

plt.xlabel('Size of bubble experiment (number of students in smaller arm)')
plt.ylabel('Frequency (number of bubbles)')
plt.hist([len(v[0]) + len(v[1]) for k, v in my_bubble_students.iteritems()])
plt.show()

plt.xlabel('Bubble path length (number of lesson interactions in shorter arm)')
plt.ylabel('Frequency (number of bubbles)')
plt.hist([min(len(path), len(other_path)) for _, _, path, other_path in my_bubble_students])
plt.show()

plt.xlabel('Bubble path length (number of lesson interactions in both arms)')
plt.ylabel('Frequency (number of bubbles)')
plt.hist([len(path) + len(other_path) for _, _, path, other_path in my_bubble_students])
plt.show()

plt.xlabel('Size of bubble experiment (number of students)')
plt.ylabel('Average bubble path length (number of lesson interactions)')
plt.scatter([min(len(v[0]), len(v[1])) for k, v in my_bubble_students.iteritems()],
            [min(len(path), len(other_path)) for _, _, path, other_path in my_bubble_students],
            alpha=0.1)
plt.show()

# instead of embedding each bubble separately
# (which would take a long time),
# assign bubbles to embedding "rounds" so that multiple 
# bubbles can be embedded simultaneously.
# in practice, the sets of students involved in different bubbles
# are disjoint enough that the resulting number of rounds
# is much lower than the number of bubbles.
unassigned_bubbles = set(my_bubble_students.keys())
rounds = []
while len(unassigned_bubbles) > 0:
    used = set()
    rd = []

    for bubble in unassigned_bubbles:
        students = set(sum(my_bubble_students[bubble], []))
        if len(students - used) < len(students):
            continue
        used |= students
        rd.append(bubble)

    rounds.append(rd)
    unassigned_bubbles -= set(rd)

print "Number of bubbles covered = %d" % sum([len(r) for r in rounds])
print "Number of rounds = %d" % len(rounds)
plt.xlabel('Round')
plt.ylabel('Number of bubbles in round')
plt.plot([len(r) for r in rounds])
plt.show()


# functions for "playing" lesson interactions over students
# and computing the expected student embedding at the end of the lessons
def play_lessons_over_student(initial_student_embedding, lesson_and_prereq_embeddings):
    """
    Simulate a student completing a sequence of lesson interactions,
    where gains are modulated by prereqs

    :param np.ndarray initial_student_embedding: The embedding of the student
        before completing any lesson interactions

    :param list[(np.ndarray, np.ndarray)] lesson_and_prereq_embeddings:
        A list of tuples (lesson embedding, prereq embedding) that contains
        the sequence of lessons the student will complete

    :rtype: np.ndarray
    :return: The expected student embedding at the end of the lesson sequence
    """
    return reduce(
        lambda prev_student_embedding,
               (lesson_embedding, prereq_embedding): prev_student_embedding + lesson_embedding / (1 + math.exp(
            -(np.dot(prev_student_embedding, prereq_embedding) / np.linalg.norm(prereq_embedding) - np.linalg.norm(
                prereq_embedding)))),
        lesson_and_prereq_embeddings, initial_student_embedding)


def play_lessons_over_student_without_prereqs(initial_student_embedding, lesson_embeddings):
    """
    Simulate a student completing a sequence of lesson interactions,
    where gains are not modulated by prereqs

    :param np.ndarray initial_student_embedding: The embedding of the student
        before completing any lesson interactions

    :param list[np.ndarray] lesson_embeddings:
        A list of embeddings for the lessons that the student will complete

    :rtype: np.ndarray
    :return: The expected student embedding at the end of the lesson sequence
    """
    return reduce(
        lambda prev_student_embedding, lesson_embedding: prev_student_embedding + lesson_embedding,
        lesson_embeddings, initial_student_embedding)


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


estimator = est.EmbeddingMAPEstimator(
    regularization_constant=1e-6,
    using_scipy=True,
    verify_gradient=False,
    debug_mode_on=True,
    ftol=1e-3)


def meta_meta_build_embedding(embedding_kwargs):
    def meta_build_embedding(
            history,
            filtered_history,
            split_history=None):
        return build_embedding(
            embedding_kwargs,
            estimator,
            history,
            filtered_history,
            split_history=split_history)

    return meta_build_embedding


embedding_kwargs = {
    'embedding_dimension': 2,
    'using_lessons': True,
    'using_prereqs': True,
    'using_bias': True,
    'learning_update_variance_constant': 0.5
}

model_builders = {
    'd=2, with prereqs and bias': meta_meta_build_embedding(embedding_kwargs)
}


def eval_embedding(
        model,
        history,
        lesson_seqs,
        timestep_of_bubble_start,
        final_assessment_id):
    """
    For students in the bubble, compute the expected pass likelihood on
    the final assessment if they take their recommended path or the alternative path

    :param models.EmbeddingModel model: A trained embedding
    :param datatools.InteractionHistory history: History used to train the model
    :param dict[str, (str, str, ...)] lesson_seqs:
        A dictionary mapping user_id to the lesson sequence of the path
        they actually took

    :param dict[str, int] timestep_of_bubble_start:
        A dictionary mapping user_id to the timestep at which that student
        worked on the start lesson for the bubble

    :param str final_assessment_id: The id of the assessment at the end of the lesson
    :param list[str] students: A list of ids for the students participating in the bubble
    :rtype: (np.array, np.array)
    :return: A tuple of (pass likelihoods )
    """

    lesson_seqs_vals = list(set(lesson_seqs.values()))

    if model.using_prereqs:
        get_lesson_seq = {v: [(
            model.lesson_embeddings[history.idx_of_lesson_id(lesson_id), :],
            model.prereq_embeddings[history.idx_of_lesson_id(lesson_id), :]) for lesson_id in v] for v in
            lesson_seqs_vals}
    else:
        get_lesson_seq = {v: [model.lesson_embeddings[history.idx_of_lesson_id(lesson_id), :] for lesson_id in v] for v
                          in lesson_seqs_vals}

    lesson_seqs_taken = {user_id: get_lesson_seq[lesson_seqs[user_id]] for user_id in lesson_seqs}

    lesson_seqs_v = {k: i for i, k in enumerate(lesson_seqs_vals)}
    lesson_seqs_nontaken = {user_id: get_lesson_seq[lesson_seqs_vals[(lesson_seqs_v[lesson_seqs[user_id]] + 1) % 2]] for
                            user_id in lesson_seqs}

    lesson_simulator = play_lessons_over_student if model.using_prereqs else play_lessons_over_student_without_prereqs

    expected_students_after_taken_path = {user_id: lesson_simulator(
        model.student_embeddings[history.idx_of_user_id(user_id), :,
        timestep_of_bubble_start[user_id]], seq) for user_id, seq in lesson_seqs_taken.iteritems()}
    expected_students_after_nontaken_path = {user_id: lesson_simulator(
        model.student_embeddings[history.idx_of_user_id(user_id), :,
        timestep_of_bubble_start[user_id]], seq) for user_id, seq in lesson_seqs_nontaken.iteritems()}

    assessment_idx = history.idx_of_assessment_id(final_assessment_id)
    assessment_embedding = model.assessment_embeddings[assessment_idx, :]
    assessment_bias = model.assessment_biases[assessment_idx]

    pass_likelihoods_after_taken_path = {user_id: math.exp(
        model.assessment_outcome_log_likelihood_helper(
            expected_student_embedding,
            assessment_embedding,
            model.student_biases[history.idx_of_user_id(user_id)],
            assessment_bias,
            True)) for user_id, expected_student_embedding in expected_students_after_taken_path.iteritems()}

    pass_likelihoods_after_nontaken_path = {user_id: math.exp(
        model.assessment_outcome_log_likelihood_helper(
            expected_student_embedding,
            assessment_embedding,
            model.student_biases[history.idx_of_user_id(user_id)],
            assessment_bias,
            True)) for user_id, expected_student_embedding in expected_students_after_nontaken_path.iteritems()}

    return pass_likelihoods_after_taken_path, pass_likelihoods_after_nontaken_path


model_evals = {
    'd=2, with prereqs and bias': eval_embedding
}

num_rounds = len(rounds)
timestep_of_last_interaction = df.groupby('user_id')['timestep'].max()

grouped_by_module = df.groupby('module_id')
# timestep_of_bubble_start[round][user_id]
# = timestep of lesson interaction at bubble start
timestep_of_bubble_start = [{} for _ in range(num_rounds)]
for i, rd in enumerate(rounds):
    for (start_lesson_id, final_assessment_id, path, other_path) in rd:
        students = sum(my_bubble_students[(start_lesson_id, final_assessment_id, path, other_path)], [])
        grouped_by_student = grouped_by_module.get_group(start_lesson_id).groupby('user_id')
        for user_id in students:
            student_group = grouped_by_student.get_group(user_id)
            timestep_of_bubble_start[i][user_id] = list(student_group['timestep'])[0]

student_pass_likelihoods = [[{k: None for k in model_builders} for _ in rd] for rd in rounds]

for i, rd in enumerate(rounds):
    print '%d of %d' % (i, num_rounds)

    t = df['user_id'].apply(
        lambda x: timestep_of_bubble_start[i][x] if x in timestep_of_bubble_start[i] else
        timestep_of_last_interaction.ix[x])

    filtered_history = df[(left_in_ixns) | ((left_out_ixns) & df['timestep'] <= t)]

    split_history = history.split_interactions_by_type(
        filtered_history=filtered_history)

    round_models = {}
    for k, build_model in model_builders.iteritems():
        round_models[k] = build_model(
            history,
            filtered_history,
            split_history=split_history)

    for j, (start_lesson_id, final_assessment_id, path, other_path) in enumerate(rd):
        students_on_path, students_on_other_path = my_bubble_students[
            (start_lesson_id, final_assessment_id, path, other_path)]
        lesson_seqs = {user_id: path for user_id in students_on_path}
        lesson_seqs.update({user_id: other_path for user_id in students_on_other_path})
        for k, eval_model in model_evals.iteritems():
            student_pass_likelihoods[i][j][k] = eval_model(
                round_models[k],
                history,
                lesson_seqs,
                timestep_of_bubble_start[i],
                final_assessment_id)

# construct feature space for students

# if we plan to use PCA to map students to a lower-dimensional feature space
# we really should construct features for all students (not just left-out students).
# in practice, there are so many students that PCA runs quite slow,
# so we add a large number of "left-in" students instead of all of them.
NUM_EXTRA_STUDENTS = 0

num_assessments = history.num_assessments()
num_modules = num_assessments + history.num_lessons()
idx_of_module_id = {module_id: idx for idx, module_id in enumerate(history.iter_assessments())}
for idx, module_id in enumerate(history.iter_lessons()):
    idx_of_module_id[module_id] = num_assessments + idx

students_in_bubbles = {user_id for v in my_bubble_students.itervalues() for user_id in sum(v, [])}
print "Number of unique students in bubbles = %d" % len(students_in_bubbles)
students_in_bubbles |= set(random.sample(history.data['user_id'].unique(), NUM_EXTRA_STUDENTS))
students_in_bubbles = {k: i for i, k in enumerate(students_in_bubbles)}

grouped = df.groupby('user_id')
X = np.zeros((len(students_in_bubbles), num_modules))
for user_id, user_idx in students_in_bubbles.iteritems():
    group = grouped.get_group(user_id)
    for module_id, outcome in zip(group['module_id'], group['outcome']):
        X[user_idx, idx_of_module_id[module_id]] = 1 if outcome is None else (1 if outcome else -1)

# map students to low-dimensional feature space using PCA
NUM_COVARIATES = 1000

pca = PCA(n_components=NUM_COVARIATES)
XS = pca.fit_transform(X)

plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.plot(pca.explained_variance_ratio_)
plt.show()

N = len(pca.explained_variance_ratio_)
x = [None] * N
x[0] = pca.explained_variance_ratio_[0]
y = range(N)
for i in range(1, N):
    x[i] = x[i - 1] + pca.explained_variance_ratio_[i]
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance ratio')
plt.plot(y, x)
plt.show()

# number of folds in k-fold cross-validation used to select
# an L2-regularization constant for logistic regression
NUM_FOLDS = 5

# regularization constants to select from
Cs = [1e-3, 1e-2, 0.1, 1.0, 10, 100]

propensity_scores = {k: {} for k in model_builders}

for model in model_builders:
    for i, rd in enumerate(rounds):
        for j, (bubble_pass_likelihoods, (start_lesson_id, final_assessment_id, path, other_path)) in enumerate(
                zip(student_pass_likelihoods[i], rd)):
            students = sum(my_bubble_students[(start_lesson_id, final_assessment_id, path, other_path)], [])
            user_idxes = np.array([students_in_bubbles[user_id] for user_id in students])
            myX = X[user_idxes, :]

            pass_likelihoods_on_taken_path, pass_likelihoods_on_nontaken_path = bubble_pass_likelihoods[model]
            Y = np.array(
                [1 if pass_likelihoods_on_taken_path[user_id] >= pass_likelihoods_on_nontaken_path[user_id] else 0 for
                 user_id in students])

            if len(set(Y)) <= 1:
                # not enough students took their recommended path
                continue

            # select L2-regularization constant using cross-validation  
            kf = KFold(
                len(students),
                n_folds=NUM_FOLDS,
                shuffle=True)

            val_lls = [[] for _ in range(len(Cs))]  # average log-likelihoods
            for k, (train_idxes, val_idxes) in enumerate(kf):
                def compute_val_ll(lreg_model):
                    """
                    Compute average log-likelihood of validation student participation
                    """
                    log_probas = [lreg_model.predict_log_proba(x) for x in myX[val_idxes]]
                    idx_of_zero = 0 if lreg_model.classes_[0] == 0 else 1
                    return np.mean([ll[0, (y_true ^ idx_of_zero)] for ll, y_true in zip(log_probas, Y[val_idxes])])


                for i, C in enumerate(Cs):
                    if len(set(Y[train_idxes])) == 1:
                        continue
                    lreg_model = LogisticRegression(penalty='l2', C=C)
                    lreg_model.fit(myX[train_idxes], Y[train_idxes])
                    val_lls[i].append(compute_val_ll(lreg_model))

            # select C that gives the highest validation average log-likelihood
            C = Cs[max(range(len(Cs)), key=lambda i: np.mean(val_lls[i]))]

            lreg_model = LogisticRegression(penalty='l2', C=C)
            lreg_model.fit(myX, Y)

            idx_of_one = 0 if lreg_model.classes_[0] == 1 else 1

            propensity_scores[model][(start_lesson_id, final_assessment_id, path, other_path)] = {
                user_id: lreg_model.predict_proba(
                    X[students_in_bubbles[user_id], :])[0, idx_of_one] for user_id in students}

# some bubbles probably got filtered
# for having too few students who took their recommended path
for k, v in propensity_scores.iteritems():
    print "%s\nNumber of bubbles with propensity scores = %d\n" % (k, len(v))


def performance_vs_path_quality_diff(
        model,
        compute_performance_metric,
        list_num_neighbors_to_match_on,
        threshold_tick_size=0.05,
        min_threshold=0.,
        max_threshold=0.4):
    """
    Generic function for computing a performance metric using
    different nearest neighbor matching and conditioning on bubbles

    :param int matching: 
        0 => no propensity score matching
        k => k-nearest neighbor matching

    :param float threshold_tick_size: Tick size for threshold on path quality difference
    :param float min_threshold: Minimum threshold for path quality difference
    :param float max_threshold: Maximum threshold for path quality difference
    :param function compute_metric:
        Compute performance on a single bubble::

            pass rate on recommended path, pass rate on non-recommended path -> metric

    :rtype: (list[list[float]], list[list[float]], list[float])
    :return: 
        Performance metrics and standard errors, 
        for each number of nearest neighbors to match on, 
        for each threshold
    """

    thresholds = np.arange(min_threshold, max_threshold, threshold_tick_size)
    performance_metrics = [[] for _ in list_num_neighbors_to_match_on]
    performance_metric_stderrs = [[] for _ in list_num_neighbors_to_match_on]

    for n, num_neighbors_to_match_on in enumerate(list_num_neighbors_to_match_on):
        for threshold in thresholds:
            performance_metrics_for_threshold = []
            # iterate over rounds
            for i, rd in enumerate(rounds):
                # iterate over bubbles in round
                for j, k in enumerate(rd):
                    start_lesson_id, final_assessment_id, path, other_path = k
                    path_students, other_path_students = my_bubble_students[k]

                    path_outcomes, other_path_outcomes = my_bubble_outcomes[k]
                    path_outcomes = {user_id: outcome for user_id, outcome in zip(path_students, path_outcomes)}
                    other_path_outcomes = {user_id: outcome for user_id, outcome in
                                           zip(other_path_students, other_path_outcomes)}
                    outcomes = {}
                    outcomes.update(path_outcomes)
                    outcomes.update(other_path_outcomes)

                    path_pass_rate = np.mean(path_outcomes.values())
                    other_path_pass_rate = np.mean(other_path_outcomes.values())

                    if abs(path_pass_rate - other_path_pass_rate) < threshold:
                        continue

                    # propensity score matching
                    if num_neighbors_to_match_on > 0:
                        # try to get propensity scores
                        try:
                            prop_scores = propensity_scores[model][k]
                        except KeyError:
                            # propensity scores not available for this bubble
                            continue

                        larger_student_group, smaller_student_group = (path_students, other_path_students) if len(
                            path_students) > len(other_path_students) else (other_path_students, path_students)

                        matched_students = set()
                        for user_id in larger_student_group:
                            ps = prop_scores[user_id]
                            matched_students |= set(sorted(
                                smaller_student_group,
                                key=lambda other_user_id: abs(
                                    prop_scores[other_user_id] - ps))[:num_neighbors_to_match_on])

                        matched_students |= set(larger_student_group)
                    else:
                        matched_students = set(path_students) | set(other_path_students)

                    outcomes = {user_id: v for user_id, v in outcomes.iteritems() if user_id in matched_students}

                    pass_likelihoods_on_taken_path, pass_likelihoods_on_nontaken_path = student_pass_likelihoods[i][j][
                        model]
                    outcomes_on_recommended_path = [1 if outcome else 0 for user_id, outcome in outcomes.iteritems() if
                                                    pass_likelihoods_on_taken_path[user_id] >=
                                                    pass_likelihoods_on_nontaken_path[user_id]]
                    outcomes_on_nonrecommended_path = [1 if outcome else 0 for user_id, outcome in outcomes.iteritems()
                                                       if pass_likelihoods_on_taken_path[user_id] <
                                                       pass_likelihoods_on_nontaken_path[user_id]]
                    if outcomes_on_recommended_path == [] or outcomes_on_nonrecommended_path == []:
                        # nobody took the recommended path, or nobody took the non-recommended path
                        continue
                    pass_rate_on_recommended_path = np.mean(outcomes_on_recommended_path)
                    pass_rate_on_nonrecommended_path = np.mean(outcomes_on_nonrecommended_path)

                    pm = compute_performance_metric(pass_rate_on_recommended_path, pass_rate_on_nonrecommended_path)
                    if not np.isinf(pm):
                        performance_metrics_for_threshold.append(pm)

            performance_metrics[n].append(np.mean(performance_metrics_for_threshold))
            performance_metric_stderrs[n].append(
                np.std(performance_metrics_for_threshold) / math.sqrt(len(performance_metrics_for_threshold)))
    return performance_metrics, performance_metric_stderrs, thresholds


def plot_performance_vs_path_quality_diff(
        model, metrics, stderrs, thresholds, labels,
        metric_name='Performance',
        random_baseline=0):
    for m, l, s in zip(metrics, labels, stderrs):
        plt.plot(thresholds, m, label=l, linewidth=3)
        plt.errorbar(thresholds, m, yerr=1.96 * np.array(s), color='black')

    plt.plot(thresholds, [random_baseline] * len(thresholds),
             '--',
             color='black',
             label='random')

    plt.xlabel('Minimum difference in path quality')
    plt.ylabel(metric_name)
    plt.title(model)
    plt.legend(loc='lower left')
    plt.legend(bbox_to_anchor=(1., 1.))
    plt.show()


model = 'd=2, with prereqs and bias'

num_neighbors_to_match_on = [0, 1, 3, 5]
labels = ['no matching'] + ['%d-NN matching' % (x) for x in num_neighbors_to_match_on[1:]]

compute_relative_gain = lambda p, q: p / (1 - p) * (1 - q) / q
metric_name = 'Expected relative gain from recommended path'

performance_metrics, performance_metric_stderrs, thresholds = performance_vs_path_quality_diff(
    model,
    compute_relative_gain,
    num_neighbors_to_match_on)

plot_performance_vs_path_quality_diff(
    model,
    performance_metrics, performance_metric_stderrs, thresholds, labels,
    metric_name=metric_name,
    random_baseline=1)

comp_sim, quality_diff = [], []
for i, rd in enumerate(rounds):
    for j, (start_lesson_id, final_assessment_id, path, other_path) in enumerate(rd):
        path_outcomes, other_path_outcomes = my_bubble_outcomes[
            (start_lesson_id, final_assessment_id, path, other_path)]
        path_pass_rate = np.mean(path_outcomes)
        other_path_pass_rate = np.mean(other_path_outcomes)
        comp_sim.append(len(set(path) & set(other_path)))
        quality_diff.append(abs(path_pass_rate - other_path_pass_rate))

plt.xlabel('Path composition similarity')
plt.ylabel('Path quality difference')
plt.scatter(comp_sim, quality_diff, alpha=0.5)
plt.show()
