import numpy as np
from operator import itemgetter

class Stats():
    def __init__(self):
        self.models = {}

    def analyse(self, stats, total):
        for type, list in stats.items():
            mean = np.mean(list)
            # print(type + ' ' + str(mean))
            print(type + ' ' + str(len(list)) + ' ' + str(int(100*len(list)/total)))

    def add_model_data(self, eval_dict, type_dict, model_name):
        self.models[model_name] = [eval_dict, type_dict]

    def summarize(self):
        print('To dooooooo')
