import numpy as np
from operator import itemgetter
from pprint import pprint
from prettytable import PrettyTable
from tabulate import tabulate


class Stats():
    def __init__(self):
        self.models = {}
        self.type_to_count_dict = None
        self.print_latex = False

    def analyse(self, stats, total):
        for type, list in stats.items():
            mean = np.mean(list)
            print(type + ' ' + str(len(list)) + ' ' + str(int(100*len(list)/total)))

    def add_model_data(self, eval_dict, type_dict, model_name):
        self.models[model_name] = [eval_dict, type_dict]

    def compare_same_results_per_type(self, model_name, same_F1, same_EM):
        id_to_type_dict = self.models[model_name][1]
        type_to_metrics_dict = {}

        for id in same_F1:
            type = id_to_type_dict[id]
            if type not in type_to_metrics_dict:
                type_to_metrics_dict[type] = {'f1':[], 'em':[]}
            type_to_metrics_dict[type]['f1'].append(id)
        for id in same_EM:
            type = id_to_type_dict[id]
            if type not in type_to_metrics_dict:
                type_to_metrics_dict[type] = {'f1':[], 'em':[]}
            type_to_metrics_dict[type]['em'].append(id)

        t = PrettyTable(['type','equal F1','equal EM','total count'])
        tabulate_table = []
        headers = ['Type','Equal F1','Equal EM','Total count']

        for type, metrics in sorted(type_to_metrics_dict.items(), key=lambda x: x[0]):
            f1_percentage = round(100*len(metrics['f1'])/self.type_to_count_dict[type],1)
            em_percentage = round(100*len(metrics['em'])/self.type_to_count_dict[type],1)
            t.add_row([type,'{} ({}%)'.format(len(metrics['f1']),f1_percentage),
                '{} ({}%)'.format(len(metrics['em']),em_percentage),
                self.type_to_count_dict[type]])
            tabulate_table.append([type,'{} ({}%)'.format(len(metrics['f1']),f1_percentage),
                '{} ({}%)'.format(len(metrics['em']),em_percentage),
                self.type_to_count_dict[type]])
        sum_f1_percentage = round(100*len(same_F1)/sum(self.type_to_count_dict.values()),1)
        sum_em_percentage = round(100*len(same_EM)/sum(self.type_to_count_dict.values()),1)
        t.add_row(['SUM','{} ({}%)'.format(len(same_F1),sum_f1_percentage),
            '{} ({}%)'.format(len(same_EM),sum_em_percentage),
            sum(self.type_to_count_dict.values())])
        tabulate_table.append(['SUM','{} ({}%)'.format(len(same_F1),sum_f1_percentage),
            '{} ({}%)'.format(len(same_EM),sum_em_percentage),
            sum(self.type_to_count_dict.values())])
        print(t)
        if self.print_latex:
            print(tabulate(tabulate_table, headers, tablefmt="latex"))

    def compare_models(self, base_model, compared_model):
        base_eval_dict = self.models[base_model][0]
        base_type_dict =  self.models[base_model][1]
        same_F1 = []
        same_EM = []
        same_F1_values = []
        same_EM_values = []
        missing = 0
        eval_dict = self.models[compared_model][0]
        for id, metrics in eval_dict.items():
            em = metrics['em']
            f1 = metrics['f1']
            if id in base_eval_dict:
                if f1 == base_eval_dict[id]['f1']:
                    same_F1.append(id)
                    same_F1_values.append(f1)
                if em == base_eval_dict[id]['em']:
                    same_EM.append(id)
                    same_EM_values.append(em)
            else:
                missing += 1

        print('\n{} ({}) vs. {} ({})'.format(base_model,str(len(base_eval_dict)),compared_model,str(len(eval_dict))))

        self.compare_same_results_per_type(base_model,same_F1,same_EM)

        print('mean of equal F1s: {}%'.format(str(round(100*np.mean(same_F1_values),1))))
        true_percentage = round(100*same_EM_values.count(True)/len(same_EM_values),1)
        print('number of Trues in equal EMs: {} ({}%)'.format(str(same_EM_values.count(True)),true_percentage))

        if missing > 0:
            print('WARNING: {} missmatched questions'.format(missing))


    def model_break_down(self, model_name):
        for name, _ in self.models.items():
            if name != model_name:
                self.compare_models(model_name, name)

    def summarize(self):
        print('\nMoldels to summarize: ' + str(len(self.models)))
        for model, _ in self.models.items():
            self.model_break_down(model)
