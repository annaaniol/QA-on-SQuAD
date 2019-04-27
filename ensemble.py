import numpy as np
from pprint import pprint
import operator


class Ensembler():
    def __init__(self):
        self.weights = None

    def get_metrics_by_type(self, stats_list, model_name_list):
        type_to_metrics_dict = {} # what -> {'mnemo':.0.8, 'qanet':0.75}
        for model in zip(stats_list, model_name_list):
            stats = model[0]
            model_name = model[1]
            for type, list_of_values in stats.items():
                if type not in type_to_metrics_dict:
                    type_to_metrics_dict[type] = {}

                type_to_metrics_dict[type][model_name] = np.mean(list_of_values)
        # pprint(type_to_metrics_dict)
        return type_to_metrics_dict

    def get_best_model_by_type(self, type_to_metrics_dict):
        type_to_leader_dict = {} # what -> 'mnemo'
        for type, dict in type_to_metrics_dict.items():
            type_to_leader_dict[type] = max(dict.items(), key=operator.itemgetter(1))[0]
        # pprint(type_to_leader_dict)
        return type_to_leader_dict

    def get_zero_one_weights(self, type_to_metrics_dict):
        zero_one_weights_dict = {} # what -> {'mnemo':1.0, 'qanet':0.0}
        for type, dict in type_to_metrics_dict.items():
            zero_one_weights_dict[type] = {}
            zero_one_weights_dict[type][max(dict.items(), key=operator.itemgetter(1))[0]] = 1.0
            for model, _ in sorted(dict.items(), key=operator.itemgetter(1), reverse=True)[1:]:
                zero_one_weights_dict[type][model] = 0.0
        pprint(zero_one_weights_dict)
        return zero_one_weights_dict

    def count_weights(self, stats_list, model_name_list, metric):
        type_to_metrics_dict = self.get_metrics_by_type(stats_list,model_name_list)
        type_to_leader_dict = self.get_best_model_by_type(type_to_metrics_dict)
        zero_one_weights_dict = self.get_zero_one_weights(type_to_metrics_dict)

        self.weights = zero_one_weights_dict

        return zero_one_weights_dict