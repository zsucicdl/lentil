from __future__ import division

import os
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('whitegrid')

from lentil import evaluate
from lentil.util import *

import logging

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

model_builders = {}

# baselines
model_builders = {
    '0PL IRT (students)': build_student_biased_coin_model,
    '0PL IRT (assessments)': build_assessment_biased_coin_model,
    '1PL IRT': build_1pl_irt_model,
    '2PL IRT': build_2pl_irt_model  # ,
    # '2D MIRT' : meta_build_mirt_model(dims=2)
}

learning_update_variances = [1e-8, 1e-6, 1e-4, 1e-2, 0.5, 10., 100., 1000.]

# vary learning_update_variance
for var in learning_update_variances:
    model_builders['d=2, with bias, var=%f' % var] = meta_build_embedding(
        d=2,
        using_lessons=True,
        using_prereqs=True,
        using_bias=True,
        learning_update_variance_constant=var)

# high learning_update_variance should simulate having no lessons
model_builders['d=2, without lessons, with bias'] = meta_build_embedding(
    d=2,
    using_lessons=False,
    using_prereqs=False,
    using_bias=True)

regularization_constants = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10.]

# vary regularization_constant
for const in regularization_constants:
    model_builders['d=2, with bias, regularization_constant=%f' % const] = meta_build_embedding(
        d=2,
        using_lessons=True,
        using_prereqs=True,
        using_bias=True,
        regularization_constant=const)

    # the effect of varying regularization 
    # is probably stronger when there are no bias terms
    model_builders['d=2, without bias, regularization_constant=%f' % const] = meta_build_embedding(
        d=2,
        using_lessons=True,
        using_prereqs=True,
        using_bias=False,
        regularization_constant=const)

# grid of regularization_constant and embedding_dimension values
embedding_dimensions = [2, 5, 10, 20, 50]
regularization_constants = [1e-8, 1e-6, 1e-4, 1e-2, 1.]

for d in embedding_dimensions:
    for const in regularization_constants:
        model_builders['d=%d, with bias, regularization_constant=%f' % (d, const)] = meta_build_embedding(
            d=d,
            using_lessons=True,
            using_prereqs=True,
            using_bias=True,
            regularization_constant=const,
            using_scipy=(d <= 10))

        # the effect of varying dimension and regularization 
        # is probably stronger when there are no bias terms
        model_builders['d=%d, without bias, regularization_constant=%f' % (d, const)] = meta_build_embedding(
            d=d,
            using_lessons=True,
            using_prereqs=True,
            using_bias=False,
            regularization_constant=const,
            using_scipy=(d <= 10))

# lesion analysis

# baselines
model_builders = {
    '0PL IRT (assessments)': build_student_biased_coin_model,
    '0PL IRT (students)': build_assessment_biased_coin_model,
    '1PL IRT': build_1pl_irt_model,
    '2PL IRT': build_2pl_irt_model,
    '2D MIRT': meta_build_mirt_model(dims=2)
}

# lesson|prereq|bias
# Y|Y|Y
model_builders['d=2, with prereqs and bias'] = meta_build_embedding(
    d=2,
    using_lessons=True,
    using_prereqs=True,
    using_bias=True)

# Y|Y|N
model_builders['d=2, with prereqs, without bias'] = meta_build_embedding(
    d=2,
    using_prereqs=True,
    using_lessons=True,
    using_bias=False)

# Y|N|N
model_builders['d=2, without prereqs and bias'] = meta_build_embedding(
    d=2,
    using_prereqs=False,
    using_lessons=True,
    using_bias=False)

# Y|N|Y
model_builders['d=2, without prereqs, with bias'] = meta_build_embedding(
    d=2,
    using_prereqs=False,
    using_lessons=True,
    using_bias=True)

# N|N|N
model_builders['d=2, without lessons and bias'] = meta_build_embedding(
    d=2,
    using_lessons=False,
    using_prereqs=False,
    using_bias=False)

# N|N|Y
model_builders['d=2, without lessons, with bias'] = meta_build_embedding(
    d=2,
    using_lessons=False,
    using_prereqs=False,
    using_bias=True)

# check how varying dimension and regularization affects MIRT
regularization_constants = [1e-2, 0.1, 1., 10.]
dimensions = [1, 2, 5, 10, 20]

# baselines
model_builders = {
    '0PL IRT (assessments)': build_student_biased_coin_model,
    '0PL IRT (students)': build_assessment_biased_coin_model,
    '1PL IRT': build_1pl_irt_model,
    '2PL IRT': build_2pl_irt_model
}

for d in dimensions:
    for r in regularization_constants:
        model_builders['%dD MIRT, with regularization_constant=%f' % (
            d, r)] = meta_build_mirt_model(regularization_constant=r, dims=d)

print "Number of models = %d" % (len(model_builders))
print '\n'.join(model_builders.keys())

results = evaluate.cross_validated_auc(
    model_builders,
    history,
    num_folds=10,
    random_truncations=False)

results_path = os.path.join(
    'results', 'last', 'lse_synthetic_results.pkl')

# dump results to file
with open(results_path, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

# load results from file, replacing current results
with open(results_path, 'rb') as f:
    results = pickle.load(f)

# load results from file, merging with current results
with open(results_path, 'rb') as f:
    results = results.merge(pickle.load(f))

# compare models to baselines
baselines = ['0PL IRT (students)', '0PL IRT (assessments)', '1PL IRT', '2PL IRT']
for baseline in baselines:
    for k in results.raw_results:
        if k == baseline:
            continue
        print "%s vs. %s:" % (k, baseline)
        print "p-value = %f" % (results.compare_validation_aucs(k, baseline))
        print ''

# compare with lessons to without lessons
print "without bias:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, without lessons and bias',
    'd=2, without prereqs and bias'))
print ''
print "with bias:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, without lessons, with bias',
    'd=2, without prereqs, with bias'))

# compare with prereqs to without prereqs
print "without bias:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, with prereqs, without bias',
    'd=2, without prereqs and bias'))
print ''
print "with bias:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, with prereqs and bias',
    'd=2, without prereqs, with bias'))

# compare with bias to without bias
print "with prereqs:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, with prereqs, without bias',
    'd=2, with prereqs and bias'))
print ''
print "without prereqs:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, without prereqs and bias',
    'd=2, without prereqs, with bias'))
print ''
print "without lessons:"
print "p-value = %f" % (results.compare_validation_aucs(
    'd=2, without lessons and bias',
    'd=2, without lessons, with bias'))

print 'Train\tValidation\tTest\tModel'
names = ['0PL IRT (students)', '0PL IRT (assessments)', '1PL IRT', '2PL IRT', 'd=2, without lessons and bias',
         'd=2, without lessons, with bias', 'd=2, without prereqs and bias', 'd=2, without prereqs, with bias',
         'd=2, with prereqs, without bias', 'd=2, with prereqs and bias']
for k in names:
    try:
        train_auc = results.training_auc_mean(k)
        val_auc = results.validation_auc_mean(k)
        test_auc = results.test_auc(k)
    except KeyError:
        continue
    print '%0.3f\t%0.3f\t\t%0.3f\t%s' % (train_auc, val_auc, test_auc, k)


def plot_d_vs_beta_grid(using_bias=True):
    with_or_without_bias = 'with' if using_bias else 'without'
    _, ax = plt.subplots()

    plt.xlabel('Embedding dimension')
    plt.ylabel('Area under ROC Curve')
    plt.title('Validation Performance')

    for const in regularization_constants:
        ax.plot(
            embedding_dimensions,
            [results.validation_auc_mean('d=%d, %s bias, regularization_constant=%f' % (
                d, with_or_without_bias, const)) for d in embedding_dimensions],
            '-s', label='beta=%f, %s bias' % (const, with_or_without_bias))

    ax.plot(embedding_dimensions, [results.validation_auc_mean('0PL IRT (students)')] * len(embedding_dimensions),
            '--', label='0PL IRT (students)')
    ax.plot(embedding_dimensions, [results.validation_auc_mean('0PL IRT (assessments)')] * len(embedding_dimensions),
            '--', label='0PL IRT (assessments)')
    ax.plot(embedding_dimensions, [results.validation_auc_mean('1PL IRT')] * len(embedding_dimensions),
            '--', label='1PL IRT')
    ax.plot(embedding_dimensions, [results.validation_auc_mean('2PL IRT')] * len(embedding_dimensions),
            '--', label='2PL IRT')

    ax.legend(loc='upper right')
    ax.legend(bbox_to_anchor=(1., 1.))

    ax.set_xscale('log')
    ax.set_xticks(embedding_dimensions)
    ax.set_xticklabels([str(x) for x in embedding_dimensions])
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    plt.show()


plot_d_vs_beta_grid(using_bias=True)

plot_d_vs_beta_grid(using_bias=False)

plt.xlabel('Regularization constant')
plt.ylabel('Area under ROC Curve')
plt.title('Validation Performance')

plt.errorbar(
    regularization_constants,
    [results.validation_auc_mean('d=2, with bias, regularization_constant=%f' % const) for const in
     regularization_constants],
    yerr=[1.96 * results.validation_auc_stderr('d=2, with bias, regularization_constant=%f' % const) for const in
          regularization_constants],
    label='d=2, with bias')

plt.errorbar(
    regularization_constants,
    [results.validation_auc_mean('d=2, without bias, regularization_constant=%f' % const) for const in
     regularization_constants],
    [1.96 * results.validation_auc_stderr('d=2, without bias, regularization_constant=%f' % const) for const in
     regularization_constants],
    label='d=2, without bias')

plt.plot(regularization_constants, [results.validation_auc_mean('0PL IRT (students)')] * len(regularization_constants),
         '--', label='0PL IRT (students)')
plt.plot(regularization_constants,
         [results.validation_auc_mean('0PL IRT (assessments)')] * len(regularization_constants),
         '--', label='0PL IRT (assessments)')
plt.plot(regularization_constants, [results.validation_auc_mean('1PL IRT')] * len(regularization_constants),
         '--', label='1PL IRT')
plt.plot(regularization_constants, [results.validation_auc_mean('2PL IRT')] * len(regularization_constants),
         '--', label='2PL IRT')

plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1., 1.))
plt.xscale('log')

plt.show()

_, ax = plt.subplots()

plt.xlabel('Learning update variance')
plt.ylabel('Area under ROC Curve')
plt.title('Validation Performance')

plt.errorbar(
    learning_update_variances,
    [results.validation_auc_mean('d=2, with bias, var=%f' % v) for v in learning_update_variances],
    yerr=[1.96 * results.validation_auc_stderr('d=2, with bias, var=%f' % v) for v in learning_update_variances],
    label='d=2, with prereqs and bias')

plt.plot(
    learning_update_variances,
    [results.validation_auc_mean('d=2, without lessons, with bias') for v in learning_update_variances],
    label='d=2, without lessons, with bias')

plt.plot(learning_update_variances,
         [results.validation_auc_mean('0PL IRT (students)')] * len(learning_update_variances),
         '--', label='0PL IRT (students)')
plt.plot(learning_update_variances,
         [results.validation_auc_mean('0PL IRT (assessments)')] * len(learning_update_variances),
         '--', label='0PL IRT (assessments)')
plt.plot(learning_update_variances, [results.validation_auc_mean('1PL IRT')] * len(learning_update_variances),
         '--', label='1PL IRT')
plt.plot(learning_update_variances, [results.validation_auc_mean('2PL IRT')] * len(learning_update_variances),
         '--', label='2PL IRT')

plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1., 1.))
plt.xscale('log')

plt.show()

_, ax = plt.subplots()

ax.set_xlabel('Dimension')
ax.set_ylabel('Area under ROC Curve')
ax.set_title('Validation Performance')

for r in regularization_constants:
    ax.plot(
        dimensions,
        [results.validation_auc_mean('%dD MIRT, with regularization_constant=%f' % (d, r)) for d in dimensions],
        '-s',
        label=('MIRT, lambda=%f' % r))

ax.plot(dimensions, [results.validation_auc_mean('0PL IRT (students)')] * len(dimensions),
        '--', label='0PL IRT (students)')
ax.plot(dimensions, [results.validation_auc_mean('0PL IRT (assessments)')] * len(dimensions),
        '--', label='0PL IRT (assessments)')
ax.plot(dimensions, [results.validation_auc_mean('1PL IRT')] * len(dimensions),
        '--', label='1PL IRT')
ax.plot(dimensions, [results.validation_auc_mean('2PL IRT')] * len(dimensions),
        '--', label='2PL IRT')

ax.legend(loc='upper right')
ax.legend(bbox_to_anchor=(1.5, 1.))
ax.set_xscale('log')
ax.set_xticks(dimensions)
ax.set_xticklabels([str(x) for x in dimensions])
ax.get_xaxis().get_major_formatter().labelOnlyBase = False
ax.set_xlim([min(dimensions), max(dimensions)])

plt.show()
