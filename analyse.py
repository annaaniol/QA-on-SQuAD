import json
from pprint import pprint
from collections import Counter
import string
import nltk
import re
import argparse
import sys
from SQuAD_metrics import SQuAD_metrics
import numpy as np

bidaf_prediction_file = 'BiDAF/prediction0-epoch7.out'
bidaf_dev_file = '../BiDAF/BiDAF-pytorch/.data/squad/dev-v1.1.json'

metrics = SQuAD_metrics()

def determine_type(question):
    question = question.lower()
    if 'how many' in question:
        type = 'how many'
    elif 'how much' in question:
        type = 'how much'
    elif 'how old' in question:
        type = 'how old'
    elif 'what' in question:
        type = 'what'
    elif 'why' in question:
        type = 'why'
    elif 'who' in question:
        type = 'who'
    elif 'how' in question:
        type = 'how'
    elif 'which' in question:
        type = 'which'
    elif 'where' in question:
        type = 'where'
    elif 'when' in question:
        type = 'when'
    elif 'what for' in question:
        type = 'what for'
    elif question.startswith('is') or question.startswith('are') or question.startswith('was') or question.startswith('were') \
     or question.startswith('has') or question.startswith('have') or question.startswith('had') or question.startswith('did') \
     or question.startswith('does') or question.startswith('do') or question.startswith('can'):
        type = 'yes-no'
    else:
        type = 'unknown'
    return type

predictions = {}
with open(bidaf_prediction_file, 'r', encoding='utf-8') as f:
    predictions = json.load(f)


ground_truths = {}
questions = {}
questions_ids = {}
with open(bidaf_dev_file, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            # context = paragraph['context']
            # tokens = word_tokenize(context)
            for qa in paragraph['qas']:
                id = qa['id']
                words = nltk.word_tokenize(qa['question'])
                # print(words)
                questions_ids[qa['question']] = id
                type = determine_type(qa['question'])
                questions[id] = (type,qa['question'])
                # question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    # s_idx = ans['answer_start']
                    # e_idx = s_idx + len(answer)

                    if id not in ground_truths:
                        ground_truths[id] = []

                    ground_truths[id].append(answer)

stats_f1 = {}
stats_em = {}
f1 = exact_match = total = 0
eval_dict = {}
for id, prediction in predictions.items():
    ground_truths_per_question = ground_truths[id]
    type = questions[id][0]
    # print(questions[id][1])

    exact_match_here = metrics.metric_max_over_ground_truths(
        metrics.exact_match_score, prediction, ground_truths_per_question)
    f1_here = metrics.metric_max_over_ground_truths(
        metrics.f1_score, prediction, ground_truths_per_question)
    exact_match += exact_match_here
    f1 += f1_here
    eval_dict[qa['id']] = {'em': exact_match_here, 'f1': f1_here}
    total += 1
    # print('F1:' + str(f1_here))
    # print('EM: ' + str(exact_match_here))

    if type in stats_f1:
        stats_f1[type].append(f1_here)
        stats_em[type].append(exact_match_here)
    else:
        stats_f1[type] = [f1_here]
        stats_em[type] = [exact_match_here]

exact_match = 100.0 * exact_match / total
f1 = 100.0 * f1 / total

print('\ntotal F1 ' + str(f1))
for type, f1_list in stats_f1.items():
    mean_f1 = np.mean(f1_list)
    print(type + ' ' + str(mean_f1))

print('\ntotal EM ' + str(exact_match))
for type, em_list in stats_em.items():
    mean_em = np.mean(em_list)
    print(type + ' ' + str(mean_em))
