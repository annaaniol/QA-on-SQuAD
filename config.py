ORIGINAL_FILES = {
    'dev_pattern_file': '../BiDAF/BiDAF-pytorch/.data/squad/dev-v1.1.json',
    'bidaf_prediction_file': 'BiDAF/prediction0-epoch1.out',
    'mnemonic_prediction_file': 'MnemonicReader/dev_full_training-m_reader.preds',
    'rnet_prediction_file': 'R-net/SQuAD-dev-v1.1-r_net.preds',
    'qanet_prediction_file': 'QANet/answers_reindexed.json',
}

ORIGINAL_CONFIG = {
    'dev_pattern_file': ORIGINAL_FILES['dev_pattern_file'],
    'models_to_process': [('Mnemonic Reader', ORIGINAL_FILES['mnemonic_prediction_file']),
    ('QANet', ORIGINAL_FILES['qanet_prediction_file']),
    ('BiDAF', ORIGINAL_FILES['bidaf_prediction_file'])]
}

CLASS_DEV_FILES = {
    'dev_pattern_file': 'data/splitted/class_dev_5.json',
    'mnemonic_prediction_file': 'MnemonicReader/splitted/class_dev_5-splitted_5.preds',
    'qanet_prediction_file': 'QANet/splitted/answers_class_dev_5_reindexed_23.json'
}

CLASS_DEV_CONFIG= {
    'dev_pattern_file': CLASS_DEV_FILES['dev_pattern_file'],
    'models_to_process': [('Mnemonic Reader', CLASS_DEV_FILES['mnemonic_prediction_file']),
    ('QANet', CLASS_DEV_FILES['qanet_prediction_file'])]
}

DEV_ON_SPLITTED_FILES = {
    'dev_pattern_file': '../BiDAF/BiDAF-pytorch/.data/squad/dev-v1.1.json',
    'qanet_prediction_file': 'QANet/splitted/dev_splitted_95_reindexed.json',
    'mnemonic_prediction_file': 'MnemonicReader/dev_95_splitted_model.preds'
}

DEV_ON_SPLITTED_CONFIG = {
    'dev_pattern_file': DEV_ON_SPLITTED_FILES['dev_pattern_file'],
    'models_to_process': [('Mnemonic Reader', DEV_ON_SPLITTED_FILES['mnemonic_prediction_file']),
    ('QANet', DEV_ON_SPLITTED_FILES['qanet_prediction_file'])]
}
