import json

dev_eval_file = 'data/dev_eval.json'
answers_file = 'log/answers.json'
result_file = 'log/reindexed.json'
evaluation_file = 'log/evaluation.json'
evaluation_result_file = 'log/evaluation_reindexed.json'

result_dict = {}
id_uuid_dict = {}
with open(dev_eval_file, 'r') as dev_file:
    with open(answers_file, 'r') as ans_file:

        dev_data = json.load(dev_file)
        ans_data = json.load(ans_file)

        for id, val in dev_data.items():
            id_uuid_dict[int(id)] = val['uuid']

        for id, answer in ans_data.items():
            result_dict[id_uuid_dict[int(id)]] = answer

with open(result_file, 'w') as outfile:
    json.dump(result_dict, outfile)

eval_result_dict = {}
with open(evaluation_file, 'r') as eval_file:
    eval_data = json.load(eval_file)
    for id, metrics in eval_data.items():
        eval_result_dict[id_uuid_dict[int(id)]] = metrics

with open(evaluation_result_file, 'w') as outfile:
    json.dump(eval_result_dict, outfile)
