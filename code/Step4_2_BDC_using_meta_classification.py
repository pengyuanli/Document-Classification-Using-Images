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

    # clf = SVC(gamma='auto', probability=True, class_weight='balanced')
    clf = SVC(kernel="linear", probability=True)
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


    # feature_selector = SelectKBest(f_classif, k=feature_no).fit(train_X, train_Y)
    #
    # train_X = feature_selector.transform(train_X)
    # test_X = feature_selector.transform(test_X)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    return train_X, train_Y, test_X, test_Y

def combine_text_image_results(data):

    doc_list = []
    methods_list = ['TA_w2v', 'C_w2v', 'IMG_pattern'] 

    for method_A in methods_list:

        for method_B in methods_list[methods_list.index(method_A)+1:]:


                print('*'*30)
                all_data = []
                all_label = []

                for doc in data['yes'].keys():
                    own_type_list = data['yes'][doc]['classifiers'].keys()
                    if (method_A in own_type_list) and (method_B in own_type_list):

                        # info_fusion = data['yes'][doc]['classifiers'][method_A]['score'] + \
                        #               data['yes'][doc]['classifiers'][method_B]['score'] # 1

                        info_fusion = data['yes'][doc]['classifiers'][method_A]['score'] + [data['yes'][doc]['classifiers'][method_A]['predict']] +\
                                      data['yes'][doc]['classifiers'][method_B]['score'] + [data['yes'][doc]['classifiers'][method_B]['predict']] +\
                                      data['yes'][doc]['classifiers'][method_C]['score'] + [data['yes'][doc]['classifiers'][method_C]['predict']] #2

                        # info_fusion = data['yes'][doc]['classifiers'][method_A]['score'] + \
                        #               data['yes'][doc]['classifiers'][method_B]['score'] + \
                        #               data['yes'][doc]['classifiers'][method_C]['score'] + \
                        #               [data['yes'][doc]['classifiers'][method_A]['predict'] or
                        #                data['yes'][doc]['classifiers'][method_B]['predict'] or
                        #                data['yes'][doc]['classifiers'][method_C]['predict']]# 3

                        # info_fusion = [data['yes'][doc]['classifiers'][method_A]['predict']] +\
                        #                 [data['yes'][doc]['classifiers'][method_B]['predict']] +\
                        #                 [data['yes'][doc]['classifiers'][method_C]['predict']]  # 4


                        # info_fusion = [data['yes'][doc]['classifiers'][method_A]['score'][1] - data['yes'][doc]['classifiers'][method_A]['score'][0]]+ \
                        #               [data['yes'][doc]['classifiers'][method_A]['predict']] + \
                        #               [data['yes'][doc]['classifiers'][method_B]['score'][1] - data['yes'][doc]['classifiers'][method_B]['score'][0]] +\
                        #               [data['yes'][doc]['classifiers'][method_B]['predict']] + \
                        #               [data['yes'][doc]['classifiers'][method_C]['score'][1] - data['yes'][doc]['classifiers'][method_C]['score'][0]] +\
                        #               [data['yes'][doc]['classifiers'][method_C]['predict']] #5

                        # info_fusion = [data['yes'][doc]['classifiers'][method_A]['score'][1] - data['yes'][doc]['classifiers'][method_A]['score'][0]]+\
                        #               [data['yes'][doc]['classifiers'][method_B]['score'][1] - data['yes'][doc]['classifiers'][method_B]['score'][0]] +\
                        #               [data['yes'][doc]['classifiers'][method_C]['score'][1] - data['yes'][doc]['classifiers'][method_C]['score'][0]] #6

                        all_data.append(info_fusion)
                        all_label.append(1)
                        doc_list.append(doc)

                for doc in data['no'].keys():
                    own_type_list = data['no'][doc]['classifiers'].keys()
                    if (method_A in own_type_list) and (method_B in own_type_list):

                        # info_fusion = data['no'][doc]['classifiers'][method_A]['score'] + \
                        #               data['no'][doc]['classifiers'][method_B]['score']# 1

                        info_fusion = data['no'][doc]['classifiers'][method_A]['score'] + [data['no'][doc]['classifiers'][method_A]['predict']] + \
                                      data['no'][doc]['classifiers'][method_B]['score'] + [data['no'][doc]['classifiers'][method_B]['predict']] + \
                                      data['no'][doc]['classifiers'][method_C]['score'] + [data['no'][doc]['classifiers'][method_C]['predict']]  # 2

                        # info_fusion = data['no'][doc]['classifiers'][method_A]['score'] + \
                        #               data['no'][doc]['classifiers'][method_B]['score'] + \
                        #               data['no'][doc]['classifiers'][method_C]['score'] + \
                        #               [data['no'][doc]['classifiers'][method_A]['predict'] or
                        #                data['no'][doc]['classifiers'][method_B]['predict'] or
                        #                data['no'][doc]['classifiers'][method_C]['predict']]  # 3

                        # info_fusion = [data['no'][doc]['classifiers'][method_A]['predict']] + \
                        #               [data['no'][doc]['classifiers'][method_B]['predict']] + \
                        #               [data['no'][doc]['classifiers'][method_C]['predict']]  # 4

                        # info_fusion = [data['no'][doc]['classifiers'][method_A]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_A]['score'][0]] + \
                        #               [data['no'][doc]['classifiers'][method_A]['predict']] + \
                        #               [data['no'][doc]['classifiers'][method_B]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_B]['score'][0]] + \
                        #               [data['no'][doc]['classifiers'][method_B]['predict']] + \
                        #               [data['no'][doc]['classifiers'][method_C]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_C]['score'][0]] + \
                        #               [data['no'][doc]['classifiers'][method_C]['predict']]  # 5

                        # info_fusion = [data['no'][doc]['classifiers'][method_A]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_A]['score'][0]] + \
                        #               [data['no'][doc]['classifiers'][method_B]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_B]['score'][0]] + \
                        #               [data['no'][doc]['classifiers'][method_C]['score'][1] -
                        #                data['no'][doc]['classifiers'][method_C]['score'][0]]  # 6

                        all_data.append(info_fusion)
                        all_label.append(0)
                        doc_list.append(doc)


                for feature_no in np.arange(1, 2, 1):
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
                        test_doc_list = [doc_list[x] for x in range(len(doc_list)) if x % 5 == fold_idx]

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

                        # for idx in range(len(test_doc_list)):
                        #     if test_doc_list[idx] in data['yes'].keys():
                        #         data['yes'][test_doc_list[idx]]['classifiers']['CombC_2']= {}
                        #         data['yes'][test_doc_list[idx]]['classifiers']['CombC_2']['predict'] = int(predictions[idx])
                        #     else:
                        #         data['no'][test_doc_list[idx]]['classifiers']['CombC_2'] = {}
                        #         data['no'][test_doc_list[idx]]['classifiers']['CombC_2']['predict'] = int(predictions[idx])

                    print(method_A + ' + ' + method_B)
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
    return data

def main():

    with open('GXD_data_vectors_classifiers_0715.json') as f:
        data = json.load(f)

    data = combine_text_image_results(data)

    # with open('GXD_data_vectors_classifiers_0615.json', 'w') as f:
    #     json.dump(data, f)

if __name__ == "__main__":
    main()
