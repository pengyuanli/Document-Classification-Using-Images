# Read text info and generate vector for them
# Adding captions
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2




def baseline_feature_creation(train_data, test_data, feature_no):
    output_training = []
    train_Y = []
    output_test = []
    test_Y = []

    for doc in train_data['yes']:
        output_training.append(train_data['yes'][doc]['vectors']['TA_average'])
        train_Y.append(1)
    for doc in train_data['no']:
        output_training.append(train_data['no'][doc]['vectors']['TA_average'])
        train_Y.append(0)


    for doc in test_data['yes']:
        output_test.append(test_data['yes'][doc]['vectors']['TA_average'])
        test_Y.append(1)
    for doc in test_data['no']:
        output_test.append(test_data['no'][doc]['vectors']['TA_average'])
        test_Y.append(0)


    scaler = StandardScaler()
    scaler.fit(output_training)

    output_training = scaler.transform(output_training)
    output_test = scaler.transform(output_test)

    # feature_selector = SelectKBest(chi2, k=int(feature_no / 2)).fit(output_training, train_Y)
    #
    # output_training = feature_selector.transform(output_training)
    # output_test = feature_selector.transform(output_test)

    return output_training, train_Y, output_test, test_Y


def svm_classification(X_train, y_train, X_test, y_test):

    clf = SVC(gamma='auto', probability=True)
    clf.fit(X_train, y_train)
    # Testing process
    y_true, y_pred = y_test, clf.predict(X_test)
    scores = clf.predict_proba(X_test)

    # print(classification_report(y_true, y_pred))
    results = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return results[0], results[1], results[2], y_pred, scores

def update_classification_result(results, test_list, predictions, scores, experiment_type):
    for i in range(len(test_list)):
        if test_list[i] in results['yes'].keys():
            results['yes'][test_list[i]]['classifiers'][experiment_type] = {}
            results['yes'][test_list[i]]['classifiers'][experiment_type]['predict'] = int(predictions[i])
            results['yes'][test_list[i]]['classifiers'][experiment_type]['score'] = [float(x) for x in scores[i]]

        if test_list[i] in results['no'].keys():
            results['no'][test_list[i]]['classifiers'][experiment_type] = {}
            results['no'][test_list[i]]['classifiers'][experiment_type]['predict'] = int(predictions[i])
            results['no'][test_list[i]]['classifiers'][experiment_type]['score'] = [float(x) for x in scores[i]]
    return results



def main():
    # load text data file
    with open('GXD_data_vectors_classifiers_0311.json') as f:
        text_data = json.load(f)

    data_classifiers = text_data

    for feature_no in np.arange(100, 201, 100):
        precision =[]
        recall = []
        fscore = []
        utility10 = []
        utility20 = []

        for fold_idx in [0, 1, 2, 3, 4]:

            train_data = {}
            test_data = {}
            train_data['yes'] = {}
            test_data['yes'] = {}
            idx = 0
            test_list = []
            for doc in text_data['yes']:
                #print(doc)
                if idx % 5 != fold_idx:
                    train_data['yes'][doc] = text_data['yes'][doc]
                if idx % 5 == fold_idx:
                    test_data['yes'][doc] = text_data['yes'][doc]
                    test_list.append(doc)
                    if 'captions' in text_data['yes'][doc].keys() and len(text_data['yes'][doc]['captions'])<1:
                        print(text_data['yes'][doc])
                idx+=1

            train_data['no'] = {}
            test_data['no'] = {}
            idx = 0
            for doc in text_data['no']:
                if idx % 5 != fold_idx:
                    train_data['no'][doc] = text_data['no'][doc]
                if idx % 5 == fold_idx:
                    test_data['no'][doc] = text_data['no'][doc]
                    test_list.append(doc)
                    if len(text_data['no'][doc]['captions'])<1:
                        print(text_data['no'][doc])
                idx += 1


            train_X, train_Y, test_X, test_Y = baseline_feature_creation(train_data, test_data, feature_no)

            p, r, f, predictions, scores = svm_classification(train_X, train_Y, test_X, test_Y)
            precision.append(p)
            recall.append(r)
            fscore.append(f)

            tp = [j for i, j in zip(test_Y, predictions) if i == 1 and j == 1]
            fp = [j for i, j in zip(test_Y, predictions) if i == 0 and j == 1]
            u10 = (10 * sum(tp) - sum(fp) )/(10 * (sum(tp)+sum(fp)))
            u20 = (20 * sum(tp) - sum(fp) )/(20 * (sum(tp)+sum(fp)))
            utility10.append(u10)
            utility20.append(u20)

            results = update_classification_result(data_classifiers, test_list, predictions, scores, 'TA_w2v')


        print('Results on ' + str(feature_no))
        print(precision)
        print(recall)
        print(fscore)
        print(utility10)
        print(utility20)
        print('precison: ', sum(precision)/len(precision), statistics.stdev(precision))
        print('recall: ', sum(recall) / len(recall), statistics.stdev(recall))
        print('fscore: ', sum(fscore) / len(fscore), statistics.stdev(fscore))
        print('-'*20)


    # with open('GXD_data_vectors_classifiers_0311.json', 'w') as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    main()