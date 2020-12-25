import argparse
import numpy as np
from sklearn.metrics import f1_score

def evaluate(prediction_labels, gt_labels):
    if len(prediction_labels) != len(gt_labels):
        return 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0

    for i in range(len(prediction_labels)):
        if prediction_labels[i] == gt_labels[i] and prediction_labels[i] == '1':
            tp += 1
        elif prediction_labels[i] == gt_labels[i] and prediction_labels[i] == '0':
            tn += 1
        elif prediction_labels[i] == '0' and gt_labels[i] == '1':
            fn += 1
        else:
            fp += 1
    if tp+fp>0 :
        precision = tp/(tp+fp)
    if tp+fn >0:
        recall = tp / (tp+fn)
    temp = f1_score(gt_labels, prediction_labels, pos_label='1')

    score = 0
    if precision+recall >0 :
        score = 2*precision*recall/(precision + recall)
    
    return score

def read_prediction_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        
    answer = []
    
    for line in lines:
        temp = line.strip()
        answer.append(temp)
        
    return answer


def read_test_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        
    answer = []
    
    for line in lines[1:]:
        temp = line.strip().split(',')
        answer.append(temp[2])
        
    return answer


def evaluation_metrics(prediction_file, test_file):
    prediction_labels = read_prediction_file(prediction_file)
    gt_labels = read_test_file(test_file)

    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction_file', type=str, default='./data/prediction.tsv')
    args.add_argument('--test_file', type=str, default='./data/answer.csv')


    config = args.parse_args()

    print(evaluation_metrics(config.prediction_file, config.test_file))