import json
import os
from shutil import copyfile
import csv
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import time
import json
from sklearn.model_selection import KFold
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
import os
import random
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import matplotlib.pyplot as plt
from shutil import copyfile
from statistics import median
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def svm_classification(X_train, y_train, X_test, y_test):

    clf = SVC(gamma='auto', probability=True)
    # clf = SVC(kernel="linear", probability=True)
    # clf = DecisionTreeClassifier(random_state=0)
    # clf = RandomForestClassifier(random_state=0)

    clf.fit(X_train, y_train)
    # Testing process
    y_true, y_pred = y_test, clf.predict(X_test)
    #scores = clf.predict_proba(X_test)

    # print(classification_report(y_true, y_pred))
    results = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return results[0], results[1], results[2], y_pred


def feature_selection(train_X, train_Y, test_X, test_Y, feature_no):


    feature_selector = SelectKBest(f_classif, k=feature_no).fit(train_X, train_Y)

    train_X = feature_selector.transform(train_X)
    test_X = feature_selector.transform(test_X)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    return train_X, train_Y, test_X, test_Y

def combine_text_image_results(data):

    doc_list = list(data['yes'].keys()) + list(data['no'].keys())
    methods_list = ['TA_average', 'C_average', 'IMG_pattern'] 

    for method_A in methods_list:

        for method_B in methods_list[methods_list.index(method_A)+1:]:

                print('*'*30)
                all_data = []
                all_label = []

                for doc in data['yes'].keys():
                    own_type_list = data['yes'][doc]['vectors'].keys()
                    if (method_A in own_type_list) and (method_B in own_type_list):

                        info_fusion = data['yes'][doc]['vectors'][method_A] + data['yes'][doc]['vectors'][method_B]# 1
                        all_data.append(info_fusion)
                        all_label.append(1)

                for doc in data['no'].keys():
                    own_type_list = data['no'][doc]['vectors'].keys()
                    if (method_A in own_type_list) and (method_B in own_type_list):

                        info_fusion = data['no'][doc]['vectors'][method_A] + data['no'][doc]['vectors'][method_B]
                        all_data.append(info_fusion)
                        all_label.append(0)



                for feature_no in np.arange(200, 301, 10):
                    precision = []
                    recall = []
                    fscore = []
                    utility10 = []
                    utility20 = []

                    for fold_idx in [0, 1, 2, 3, 4]:

                        train_X = [all_data[x] for x in range(len(all_data)) if x% 5 != fold_idx]
                        train_Y = [all_label[x] for x in range(len(all_data)) if x% 5 != fold_idx]
                        test_X = [all_data[x] for x in range(len(all_data)) if x% 5 == fold_idx]
                        test_Y = [all_label[x] for x in range(len(all_data)) if x% 5 == fold_idx]

                        train_X, train_Y, test_X, test_Y = feature_selection(train_X, train_Y, test_X, test_Y, feature_no)

                        p, r, f, predictions = svm_classification(train_X, train_Y, test_X, test_Y)
                        precision.append(p)
                        recall.append(r)
                        fscore.append(f)

                        tp = [j for i, j in zip(test_Y, predictions) if i == 1 and j == 1]
                        fp = [j for i, j in zip(test_Y, predictions) if i == 0 and j == 1]
                        u10 = (10 * sum(tp) - sum(fp)) / (10 * (sum(tp) + sum(fp)))
                        u20 = (20 * sum(tp) - sum(fp)) / (20 * (sum(tp) + sum(fp)))
                        utility10.append(u10)
                        utility20.append(u20)

                    print(method_A + ' + ' + method_B )
                    print(feature_no)
                    print(precision)
                    print(recall)
                    print(fscore)

                    print(utility10)
                    print(utility20)

                    print('precsion: ', sum(precision) / len(precision), statistics.stdev(precision))
                    print('recall: ', sum(recall) / len(recall), statistics.stdev(recall))
                    print('fscore: ', sum(fscore) / len(fscore), statistics.stdev(fscore))
                    print('-' * 20)


def main():

    with open('GXD_data_vectors_classifiers_0715.json') as f:
        data = json.load(f)

    combine_text_image_results(data)



if __name__ == "__main__":
    main()
