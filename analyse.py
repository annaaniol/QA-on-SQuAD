import json
from pprint import pprint
from collections import Counter
import string
import nltk
import re
import argparse
import sys
from SQuAD_metrics import SQuAD_metrics
from plot import Plotter
from stats import Stats
import numpy as np
from collections import OrderedDict
from operator import itemgetter

metrics = SQuAD_metrics()
stats = Stats()

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

def analyze_model(model_name, model_prediction_file, dev_pattern_file):
    predictions = {}
    with open(model_prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    ground_truths = {}
    questions = {}
    questions_ids = {}
    with open(dev_pattern_file, 'r', encoding='utf-8') as f:
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
    id_to_type_dict = {}
    for id, prediction in predictions.items():
        ground_truths_per_question = ground_truths[id]
        type = questions[id][0]
        id_to_type_dict[id] = type

        exact_match_here = metrics.metric_max_over_ground_truths(
            metrics.exact_match_score, prediction, ground_truths_per_question)
        f1_here = metrics.metric_max_over_ground_truths(
            metrics.f1_score, prediction, ground_truths_per_question)
        exact_match += exact_match_here
        f1 += f1_here
        eval_dict[id] = {'em': exact_match_here, 'f1': f1_here}
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

    stats_f1 = OrderedDict(sorted(stats_f1.items(), key = itemgetter(0), reverse = False))
    stats_em = OrderedDict(sorted(stats_em.items(), key = itemgetter(0), reverse = False))

    print('\n' + str(model_name) + ' ' + str(total) + ' evaluation questions')
    print('total F1 ' + str(f1))
    stats.add_model_data(eval_dict,id_to_type_dict,model_name)
    # for type, f1_list in stats_f1.items():
    #     mean_f1 = np.mean(f1_list)
    #     print(type + ' ' + str(mean_f1))
    #     print(type + ' ' + str(len(f1_list)) + ' ' + str(int(100*len(f1_list)/total)))

    print('total EM ' + str(exact_match))
    # for type, em_list in stats_em.items():
    #     mean_em = np.mean(em_list)
    #     print(type + ' ' + str(mean_em))
    #     print(type + ' ' + str(len(em_list)) + ' ' + str(int(100*len(em_list)/total)))

    result_dict = {}
    result_dict['f1'] = [stats_f1, f1]
    result_dict['em'] = [stats_em, exact_match]
    return result_dict


def count_question_types(dev_pattern_file):
    type_to_count_dict = {}
    with open(dev_pattern_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
        for article in data:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    type = determine_type(qa['question'])
                    if type in type_to_count_dict:
                        type_to_count_dict[type] += 1
                    else:
                        type_to_count_dict[type] = 1
    return type_to_count_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', default='original')
    parser.add_argument('--percentage', dest='percentage', default='0')
    args = parser.parse_args()
    type = args.type
    percentage = args.percentage

    stats_f1_list = []
    f1_list = []
    stats_em_list = []
    em_list = []
    names_list = []
    plotter = Plotter(type,percentage)

    if type == 'original':
        dev_pattern_file = '../BiDAF/BiDAF-pytorch/.data/squad/dev-v1.1.json'
        bidaf_prediction_file = 'BiDAF/prediction0-epoch7.out'
        mnemonic_prediction_file = 'MnemonicReader/SQuAD-dev-v1.1-m_reader.preds'
        rnet_prediction_file = 'R-net/SQuAD-dev-v1.1-r_net.preds'
        qanet_prediction_file = 'QANet/answers_reindexed.json'

        models_to_process = [('R-net', rnet_prediction_file), ('Mnemonic Reader', mnemonic_prediction_file), ('QANet', qanet_prediction_file)]
    elif type == 'splitted':
        dev_pattern_file = 'data/splitted/class_dev_{}.json'.format(str(int(percentage)))
        mnemonic_prediction_file = 'MnemonicReader/splitted/class_dev_{}-m_reader.preds'.format(str(int(percentage)))

        models_to_process = [('Mnemonic Reader', mnemonic_prediction_file)]
    else:
        print('type must be original or splitted')
        sys.exit(1)

    stats.type_to_count_dict = count_question_types(dev_pattern_file)

    for model in models_to_process:
        name = model[0]
        file = model[1]
        results = analyze_model(name, file, dev_pattern_file)
        stats_f1_list.append(results['f1'][0])
        f1_list.append(results['f1'][1])
        stats_em_list.append(results['em'][0])
        em_list.append(results['em'][1])
        names_list.append(name)

    stats.summarize()

    plotter.plot_bar(stats_f1_list, f1_list, names_list, 'F1')
    plotter.plot_bar(stats_em_list, em_list, names_list, 'EM')

if __name__ == '__main__':
    main()
