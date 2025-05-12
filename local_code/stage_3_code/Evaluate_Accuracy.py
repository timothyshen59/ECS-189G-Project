'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print("Evaluating")
        accuracy = accuracy_score(self.data['true_y'], self.data['pred_y'])
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        precision = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)
        recall = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        return accuracy, f1, precision, recall
