# %%

from __future__ import division

import os

import numpy as np
import pandas as pd
import pygraphviz as pgv
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('whitegrid')

from lentil.util import *

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


print '\n'.join(df.columns)

print "Number of interactions = %d" % (len(df))
print "Number of unique students = %d" % (len(df['user_id'].unique()))
print "Number of unique modules = %d" % (len(df['problem_id'].unique()))

plt.xlabel('Number of interactions per student')
plt.ylabel('Frequency (number of students)')
plt.hist(df['user_id'].value_counts().values)
plt.show()

plt.xlabel('Number of interactions per problem')
plt.ylabel('Frequency (number of problems)')
plt.hist(df['problem_id'].value_counts().values)
plt.show()

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
num_interactions = len(df)
value_counts = df['module_type'].value_counts()
# num_assessment_ixns = value_counts.get('test_mode', 0)
num_assessment_ixns = value_counts.get(datatools.AssessmentInteraction.MODULETYPE, 0)
# num_lesson_ixns = value_counts.get('tutor', 0)
num_lesson_ixns = value_counts.get(datatools.LessonInteraction.MODULETYPE, 0)

print "Number of interactions = %d" % (num_interactions)
print "Number of assessment interactions = %d" % (num_assessment_ixns)
print "Number of lesson interactions = %d" % (num_lesson_ixns)

num_students = history.num_students()

print "Number of unique students: %d" % (num_students)

num_assessments = history.num_assessments()

print "Number of unique assessments: %d" % (num_assessments)

num_lessons = history.num_lessons()

print "Number of unique lessons: %d" % (num_lessons)

value_counts = df['outcome'].value_counts()
num_passes = value_counts.get(True, 0)
num_fails = value_counts.get(False, 0)
pass_rate = num_passes / (num_passes + num_fails)

print "Overall pass rate: %f" % (pass_rate)

d = []
for _, group in df.groupby('user_id'):
    d.extend(group['timestep'].value_counts().values)
d = np.array(d) - 1  # remove the lesson interaction at each timestep

plt.xlabel('Number of assessment interactions per timestep')
plt.ylabel('Frequency (number of timesteps)')
plt.hist(d)
plt.show()

timestamps = pd.DatetimeIndex(df['timestamp'])

print "Beginning of data set = %s" % (min(timestamps))
print "End of data set = %s" % (max(timestamps))

# hours = timestamps.hour
# plt.xlabel('Hour of interaction')
# plt.ylabel('Frequency (number of interactions)')
# plt.hist(hours, bins=24)
# plt.show()
#
# # Monday=0, Sunday=6
# days = timestamps.weekday
# plt.xlabel('Day of interaction')
# plt.ylabel('Frequency (number of interactions)')
# plt.hist(days, bins=7)
# plt.show()
#
# plt.xlabel('Timestep')
# plt.ylabel('Frequency (number of interactions)')
# plt.hist(df['timestep'].values)
# plt.show()

durations = np.array([x for x in df['duration'].values])

plt.xlabel('ln(response time, in seconds)')
plt.ylabel('Frequency (number of interactions)')
plt.hist(np.log(durations + 1))
plt.show()

counts = df['user_id'].value_counts().values
plt.xlabel('Number of interactions per student')
plt.ylabel('Frequency (number of students)')
plt.hist(counts)
plt.show()

counts = df['module_id'][df['module_type'] == datatools.LessonInteraction.MODULETYPE].value_counts().values

plt.xlabel('Number of interactions per lesson module')
plt.ylabel('Frequency (number of lesson modules)')
plt.hist(counts)
plt.show()

counts = df['module_id'][df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].value_counts().values

plt.xlabel('Number of interactions per assessment module')
plt.ylabel('Frequency (number of assessment modules)')
plt.hist(counts)
plt.show()

counts = df.groupby(['user_id', 'module_id']).size().values

plt.xlabel('Number of interactions per student per module')
plt.ylabel('Frequency (number of student-module pairs)')
plt.hist(counts)
plt.show()

num_students_per_module = [len(group['user_id'].unique()) for _, group in df.groupby('module_id')]

plt.xlabel('Number of students per module')
plt.ylabel('Frequency (number of modules)')
plt.hist(num_students_per_module)
plt.show()

grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('user_id')
num_assessments_per_student = [len(group['module_id']) for _, group in grouped]

plt.xlabel('Number of assessment modules per student')
plt.ylabel('Frequency (number of students)')
plt.hist(num_assessments_per_student)
plt.show()

grouped = df[df['module_type'] == datatools.LessonInteraction.MODULETYPE].groupby('user_id')
num_lessons_per_student = [len(group['module_id']) for _, group in grouped]

plt.xlabel('Number of lesson modules per student')
plt.ylabel('Frequency (number of students)')
plt.hist(num_lessons_per_student)
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


grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('user_id')

plt.xlabel('Student pass rate')
plt.ylabel('Frequency (number of students)')
plt.hist(get_pass_rates(grouped).values())
plt.show()

grouped = df[df['module_type'] == datatools.AssessmentInteraction.MODULETYPE].groupby('module_id')

plt.xlabel('Assessment pass rate')
plt.ylabel('Frequency (number of assessments)')
plt.hist(get_pass_rates(grouped).values())
plt.show()


def make_flow_graph(interaction_logs):
    """
    Create a graphviz object for the graph of 
    module transitions across all student paths

    :param pd.DataFrame interaction_logs: An interaction history
    :rtype pgv.AGraph
    :return Graph of module transitions in student paths
    """
    G = pgv.AGraph(directed=True)

    for module_id in interaction_logs['module_id'].unique():
        G.add_node(module_id)

    E = defaultdict(set)
    grouped = interaction_logs.groupby('user_id')
    for user_id, group in grouped:
        module_ids_in_student_path = group['module_id']
        for source_node, target_node in zip(module_ids_in_student_path[:-1], module_ids_in_student_path[1:]):
            if source_node != target_node:  # stationary
                E[(source_node, target_node)] |= {user_id}

    for (source_node, target_node), students_that_made_transition in E.iteritems():
        G.add_edge(
            source_node,
            target_node,
            weight=len(students_that_made_transition))

    return G


G = make_flow_graph(df)

flow_graph_path = os.path.join('data', 'assistments_flow_graph.dot')

G.write(flow_graph_path)


def make_conn_graph(interaction_logs):
    """
    Create a graphviz object for the bi-partite graph connecting students
    with the modules they've interacted with

    :param pd.DataFrame interaction_logs: An interaction history
    :rtype pgv.AGraph
    :return Bi-partite graph of student-module interactions
    """
    G = pgv.AGraph(directed=True)

    for module_id in interaction_logs['module_id'].unique():
        G.add_node(module_id, label='module')

    grouped = interaction_logs.groupby('user_id')
    for user_id, group in grouped:
        G.add_node(user_id, label='student')
        for module_id in set(group['module_id'].values):
            G.add_edge(user_id, module_id)

    return G


G = make_conn_graph(df)

conn_graph_path = '/home/zvonimir/PycharmProjects/lentil/data/assistments_flow_graph.dot'

G.write(conn_graph_path)
