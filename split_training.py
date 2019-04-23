import json
import argparse
from pprint import pprint
import os, sys


def split(percentage, original_train, new_train, class_dev):
    train_dict = {}
    with open(original_train, 'r') as f:
        train_data = json.load(f)['data']

    train_count = len(train_data)
    class_dev_count = int(percentage/100 * train_count)
    new_train_count = int(train_count - class_dev_count)

    new_train_data = train_data[:new_train_count]
    class_dev_data = train_data[new_train_count:]

    with open(new_train, 'w') as f:
        json.dump({'data':new_train_data}, f)

    with open(class_dev, 'w') as f:
        json.dump({'data':class_dev_data}, f)

    new_train_questions = 0
    for element in new_train_data:
        context_list = element['paragraphs']
        for el in context_list:
            for qas in el['qas']:
                new_train_questions += len(qas)

    class_dev_questions = 0
    for element in class_dev_data:
        context_list = element['paragraphs']
        for el in context_list:
            for qas in el['qas']:
                class_dev_questions += len(qas)


    print('new_train (size {}/{}): {}'.format(str(len(new_train_data)),str(new_train_questions),new_train))
    print('class_dev (size {}/{}): {}'.format(str(len(class_dev_data)),str(class_dev_questions),class_dev))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, dest='percentage')
    parser.add_argument('--train_dir', dest='train_dir')
    parser.add_argument('--original_train', dest='original_train')
    parser.add_argument('--new_folder', dest='new_folder', default='')

    args = parser.parse_args()
    percentage = args.percentage
    train_dir = args.train_dir
    original_train = args.train_dir+'/'+args.original_train
    new_folder = args.new_folder

    if not os.path.exists(train_dir+'/'+new_folder):
        os.mkdir(train_dir+'/'+new_folder)

    if len(new_folder) > 0:
        new_train = "{}/{}/train_{}.json".format(train_dir,new_folder,str(int(100-percentage)))
        class_dev = "{}/{}/class_dev_{}.json".format(train_dir,new_folder,str(int(percentage)))
    else:
        new_train = "{}/train_{}.json".format(train_dir,str(100-percentage))
        class_dev = "{}/class_dev_{}.json".format(train_dir,str(percentage))

    split(percentage,original_train,new_train,class_dev)

if __name__ == '__main__':
    main()
