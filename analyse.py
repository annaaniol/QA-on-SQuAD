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
from ensemble import Ensembler
from stats import Stats
import numpy as np
from collections import OrderedDict
from operator import itemgetter
from prettytable import PrettyTable
from tabulate import tabulate
import config

class Analyser():
    def __init__(self, ensembler):
        self.metrics = SQuAD_metrics()
        self.stats = Stats()
        self.ensembler = Ensembler()
        self.id_to_type_dict = {}

    def determine_type(self, question):
        question = question.lower()

        if question.startswith('how many') \
            or question.startswith('how much'):
            type = 'how m/m'
        #
        # elif 'how many' in question \
        #     or 'how much' in question:
        #     type = 'how m/m'

        elif 'what time' in question:
            type = 'what time'
        elif 'how big' in question in question or 'what size' in question or 'what is the size' in question:
            type = 'how big/size'
        elif 'during' in question:
            type = 'during'
        elif 'whom' in question:
            type = 'whom'

        elif question.startswith('how old') or ', how old' in question:
            type = 'how old'
        elif question.startswith('why') or ', why' in question:
            type = 'why'
        elif question.startswith('who') or ', who' in question:
            type = 'who'
        elif question.startswith('how are'):
            type = 'how are'
        elif question.startswith('in which') or question.startswith('at which') or question.startswith('which') or ', which' in question:
            type = 'which'
        elif question.startswith('where is') or question.startswith('where are'):
            type = 'where'
        elif question.startswith('when') or ', when' in question:
            type = 'when'
        elif question.startswith('at what') or question.startswith('in what') or question.startswith('what') or 'what' in question:
            type = 'what'
        else:
            # print(question)
            type = 'unknown'

        return type

    def analyze_model(self, model_name, model_prediction_file, dev_pattern_file):
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
                        type = self.determine_type(qa['question'])
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
            self.id_to_type_dict[id] = type

            exact_match_here = self.metrics.metric_max_over_ground_truths(
                self.metrics.exact_match_score, prediction, ground_truths_per_question)
            f1_here = self.metrics.metric_max_over_ground_truths(
                self.metrics.f1_score, prediction, ground_truths_per_question)
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
        print('total EM ' + str(exact_match))
        self.stats.add_model_data(eval_dict,self.id_to_type_dict,model_name)

        result_dict = {}
        result_dict['f1'] = [stats_f1, f1]
        result_dict['em'] = [stats_em, exact_match]
        return result_dict


    def count_question_types(self, dev_pattern_file, print_latex):
        type_to_count_dict = {}
        with open(dev_pattern_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
            for article in data:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        type = self.determine_type(qa['question'])
                        if type in type_to_count_dict:
                            type_to_count_dict[type] += 1
                        else:
                            type_to_count_dict[type] = 1

        pretty_table = PrettyTable(['Type','Total count','Percentage'])
        tabulate_table = []
        headers = ['Type','Total count','Percentage']
        for type, count in sorted(type_to_count_dict.items(), key=lambda x: x[0]):
            percentage = round(100 * count / sum(type_to_count_dict.values()),1)
            pretty_table.add_row([type,count,'{}%'.format(percentage)])
            tabulate_table.append([type,count,'{}%'.format(percentage)])
        pretty_table.add_row(['SUM',sum(type_to_count_dict.values()),'100%'])
        tabulate_table.append(['SUM',sum(type_to_count_dict.values()),'100%'])
        print(pretty_table)
        if print_latex:
            print(tabulate(tabulate_table, headers, tablefmt="latex"))
        return type_to_count_dict

    def get_candidate_predictions(self, models_to_process):
        candidate_predictions = {}
        for name, prediction_file in models_to_process:
            with open(prediction_file) as f:
                predictions_per_model = json.load(f)
            candidate_predictions[name] = predictions_per_model
        return candidate_predictions

    def get_id_to_type_dict(self, dev_pattern_file):
        id_to_type = {}
        with open(dev_pattern_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
            for article in data:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        type = self.determine_type(qa['question'])
                        id_to_type[qa['id']] = type
        return id_to_type

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--type', dest='type', default='original')
        parser.add_argument('--percentage', dest='percentage', default='5')
        parser.add_argument('--latex', dest='latex', default=False)
        args = parser.parse_args()
        type = args.type
        percentage = args.percentage
        print_latex = args.latex

        stats_f1_list = []
        f1_list = []
        stats_em_list = []
        em_list = []
        names_list = []
        plotter = Plotter(type,percentage)

        if type == 'original':
            dev_pattern_file =  config.ORIGINAL_CONFIG['dev_pattern_file']
            models_to_process = config.ORIGINAL_CONFIG['models_to_process']
        elif type == 'class_dev': # preds on 5% of training (pre-evaluation), trained with splitted training
            dev_pattern_file = config.CLASS_DEV_CONFIG['dev_pattern_file']
            models_to_process = config.CLASS_DEV_CONFIG['models_to_process']
        elif type == 'dev_on_splitted': # preds on original dev, trained with splitted training
            dev_pattern_file =  config.DEV_ON_SPLITTED_CONFIG['dev_pattern_file']
            models_to_process = config.DEV_ON_SPLITTED_CONFIG['models_to_process']
        elif type == 'ensemble':
            # original dev to construct id_to_type_dict
            print('\n 1. Step: original dev to construct id_to_type_dict\n')
            dev_pattern_file = config.ORIGINAL_CONFIG['dev_pattern_file']
            id_to_type = self.get_id_to_type_dict(dev_pattern_file)
            for k,v in id_to_type.items():
                self.id_to_type_dict[k] = v

            # class_dev to obtain weights
            print('\n 2. Step: class_dev to obtain weights\n')
            dev_pattern_file = config.CLASS_DEV_CONFIG['dev_pattern_file']
            models_to_process = config.CLASS_DEV_CONFIG['models_to_process']
            self.stats.type_to_count_dict = self.count_question_types(dev_pattern_file,print_latex)
            self.stats.print_latex = print_latex

            for model in models_to_process:
                name = model[0]
                file = model[1]
                results = self.analyze_model(name, file, dev_pattern_file)
                stats_f1_list.append(results['f1'][0])
                f1_list.append(results['f1'][1])
                stats_em_list.append(results['em'][0])
                em_list.append(results['em'][1])
                names_list.append(name)

            # self.stats.summarize()
            plotter.plot_bar(stats_f1_list, f1_list, names_list, 'F1', 'class_dev')
            plotter.plot_bar(stats_em_list, em_list, names_list, 'EM', 'class_dev')

            weights = self.ensembler.count_weights(stats_f1_list, names_list, 'F1')

            # dev_on_splitted to get candidate answers
            print('\n 3. Step: dev_on_splitted to get candidate answers\n')
            models_to_process = config.ORIGINAL_CONFIG['models_to_process']
            candidate_predictions = self.get_candidate_predictions(models_to_process)

            # ensemble.predict to get ensemble answers -> save to file
            print('\n 4. Step: ensemble.predict to get ensemble answers -> save to file\n')
            ensemble_predictions = self.ensembler.predict(candidate_predictions, self.id_to_type_dict, weights)
            with open(config.ENSEMBLE_FILE, 'w') as f:
                json.dump(ensemble_predictions,f)

            # evaluate ensemble predictions (vs. 100% of training results)
            # ??? vs. splitted or full training
            print('\n 5. Step: evaluate ensemble predictions (vs. 100% training results)\n')
            dev_pattern_file =  config.ORIGINAL_CONFIG['dev_pattern_file']
            models_to_process = config.ORIGINAL_CONFIG['models_to_process']
            models_to_process.append(('Ensemble',config.ENSEMBLE_FILE))
            print(models_to_process)

            stats_f1_list = []
            f1_list = []
            stats_em_list = []
            em_list = []
            names_list = []
            for model in models_to_process:
                name = model[0]
                print('\nAnalysing {}...'.format(name))
                file = model[1]
                results = self.analyze_model(name, file, dev_pattern_file)
                stats_f1_list.append(results['f1'][0])
                f1_list.append(results['f1'][1])
                stats_em_list.append(results['em'][0])
                em_list.append(results['em'][1])
                names_list.append(name)

            # self.stats.summarize()

            plotter.type = 'ensemble'
            plotter.plot_bar(stats_f1_list, f1_list, names_list, 'F1', type)
            plotter.plot_bar(stats_em_list, em_list, names_list, 'EM', type)

        else:
            print('type must be original, class_dev, dev_on_splitted or ensemble')
            sys.exit(1)

        self.stats.type_to_count_dict = self.count_question_types(dev_pattern_file,print_latex)
        self.stats.print_latex = print_latex

        if type != 'ensemble':
            for model in models_to_process:
                name = model[0]
                print('\nAnalysing {}...'.format(name))
                file = model[1]
                results = self.analyze_model(name, file, dev_pattern_file)
                stats_f1_list.append(results['f1'][0])
                f1_list.append(results['f1'][1])
                stats_em_list.append(results['em'][0])
                em_list.append(results['em'][1])
                names_list.append(name)

            self.stats.summarize()

            plotter.plot_bar(stats_f1_list, f1_list, names_list, 'F1', type)
            plotter.plot_bar(stats_em_list, em_list, names_list, 'EM', type)

if __name__ == '__main__':
    ensembler = Ensembler()
    analyser = Analyser(ensembler)

    analyser.main()
