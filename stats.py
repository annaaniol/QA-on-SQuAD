import numpy as np
from operator import itemgetter
from pprint import pprint

class Stats():
    def __init__(self):
        self.models = {}

    def analyse(self, stats, total):
        for type, list in stats.items():
            mean = np.mean(list)
            print(type + ' ' + str(len(list)) + ' ' + str(int(100*len(list)/total)))

    def add_model_data(self, eval_dict, type_dict, model_name):
        self.models[model_name] = [eval_dict, type_dict]

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
        print('same F1: {}'.format(str(len(same_F1))))
        print('average same F1: {}'.format(str(np.mean(same_F1_values))))
        print('same EM: {}'.format(str(len(same_EM))))
        print('same Trues: {}'.format(str(same_EM_values.count(True))))

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
