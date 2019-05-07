import numpy as np
from pprint import pprint
import operator
from SQuAD_metrics import SQuAD_metrics
import config
from operator import itemgetter


class Ensembler():
    def __init__(self):
        self.SQuAD_metrics = SQuAD_metrics()
        pass

    def get_metrics_by_type(self, stats_list, model_name_list):
        type_to_metrics_dict = {} # what -> {'mnemo':.0.8, 'qanet':0.75}
        for model in zip(stats_list, model_name_list):
            stats = model[0]
            model_name = model[1]
            for type, list_of_values in stats.items():
                if type not in type_to_metrics_dict:
                    type_to_metrics_dict[type] = {}

                type_to_metrics_dict[type][model_name] = np.mean(list_of_values)
        pprint(type_to_metrics_dict)
        return type_to_metrics_dict

    def get_best_model_by_type(self, type_to_metrics_dict):
        type_to_leader_dict = {} # what -> 'mnemo'
        for type, dict in type_to_metrics_dict.items():
            type_to_leader_dict[type] = max(dict.items(), key=operator.itemgetter(1))[0]
        return type_to_leader_dict

    def get_proportional_weights(self, type_to_metrics_dict):
        proportional_weights_dict = {} # what -> {'mnemo':0.812233, 'qanet':0.79234}
        for type, dict in type_to_metrics_dict.items():
            proportional_weights_dict[type] = {}
            for model, metric_score in dict.items():
                proportional_weights_dict[type][model] = metric_score
        return proportional_weights_dict

    def get_zero_one_weights(self, type_to_metrics_dict):
        zero_one_weights_dict = {} # what -> {'mnemo':1.0, 'qanet':0.0}
        for type, dict in type_to_metrics_dict.items():
            zero_one_weights_dict[type] = {}
            zero_one_weights_dict[type][max(dict.items(), key=operator.itemgetter(1))[0]] = 1.0
            for model, _ in sorted(dict.items(), key=operator.itemgetter(1), reverse=True)[1:]:
                zero_one_weights_dict[type][model] = 0.0
        return zero_one_weights_dict

    def count_weights(self, stats_list, model_name_list, metric):
        type_to_metrics_dict = self.get_metrics_by_type(stats_list,model_name_list)
        type_to_leader_dict = self.get_best_model_by_type(type_to_metrics_dict)
        weights_dict = self.get_proportional_weights(type_to_metrics_dict)

        return weights_dict

    def update_undefined_type_weight(self, weights_dict, names_list, f1_list):
        zipped = zip(names_list,f1_list)
        highest_f1_model_name = max(zipped,key=itemgetter(1))[0]

        weights_dict[config.UNDEFINED_TYPE][highest_f1_model_name] = 1.0
        return weights_dict

    def predict(self, candidate_predictions, id_to_type_dict, weights):
        # candidate_predictions = {model1: {id1: pred1, id2: pred2}, model2: {...}}
        # weights = {type1: {model1: weight1, model2: weight}, type2: {...}}
        id_to_predictions_dict = {} # id -> {pred1: 0.8, pred2: 03}

        for model_name, predictions_dict in candidate_predictions.items():
            for id, prediction in predictions_dict.items():
                if id not in id_to_predictions_dict:
                    id_to_predictions_dict[id] = {}
                question_type = id_to_type_dict[id]
                if prediction in id_to_predictions_dict[id].keys():
                    id_to_predictions_dict[id][prediction] += \
                        weights[question_type][model_name]
                else:
                    id_to_predictions_dict[id][prediction] = weights[question_type][model_name]

        ensemble_predictions = {}
        for id, predictions in id_to_predictions_dict.items():
            predictions_list = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)
            ensemble_predictions[id] = predictions_list[0][0]
        return ensemble_predictions
