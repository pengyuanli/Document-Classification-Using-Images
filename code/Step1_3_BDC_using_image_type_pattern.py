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
import os
import random
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import matplotlib.pyplot as plt
from shutil import copyfile
from statistics import median
from collections import Counter



def feature_creation(train_data, test_data, feature_no, feature_list):


    output_training = []
    output_training_image = []
    train_Y = []
    output_test = []
    output_test_image = []
    test_Y = []

    for doc in train_data['yes']:
        output_training_image.append(create_pattern_feature(train_data['yes'][doc], feature_list))
        train_Y.append(1)
    for doc in train_data['no']:
        output_training_image.append(create_pattern_feature(train_data['no'][doc], feature_list))
        train_Y.append(0)


    for doc in test_data['yes']:
        output_test_image.append(create_pattern_feature(test_data['yes'][doc], feature_list))
        test_Y.append(1)
    for doc in test_data['no']:
        output_test_image.append(create_pattern_feature(test_data['no'][doc], feature_list))
        test_Y.append(0)


    # output_training = vector_generator.transform(output_training)
    # output_test = vector_generator.transform(output_test)
    #
    # feature_selector = SelectKBest(chi2, k=feature_no).fit(output_training, train_Y)
    # output_training = feature_selector.transform(output_training)
    # output_test = feature_selector.transform(output_test)

    feature_selector = SelectKBest(chi2, k=feature_no).fit(output_training_image, train_Y)
    output_training_image = feature_selector.transform(output_training_image)
    output_test_image = feature_selector.transform(output_test_image)

    scaler = StandardScaler()
    scaler.fit(output_training_image)
    output_training_image = scaler.transform(output_training_image)
    output_test_image = scaler.transform(output_test_image)


    # #
    # output_training = np.hstack((output_training.toarray(), output_training_image))
    # output_test = np.hstack((output_test.toarray(), output_test_image))

    return output_training_image, train_Y, output_test_image, test_Y

def create_pattern_feature(doc_features, feature_list):
    vector = [0]*len(feature_list)
    for img in doc_features['imgs']:

        type = convert_binary_to_int(doc_features['imgs'][img]['type_pattern'])
        if type in feature_list:
            vector[feature_list.index(type)] += 1
    # print(vector)
    return vector # doc_features['type_feature'] +


def svm_classification(X_train, y_train, X_test, y_test):

    clf = SVC(gamma='auto', probability=True)
    clf.fit(X_train, y_train)
    # Testing process
    y_true, y_pred = y_test, clf.predict(X_test)
    scores = clf.predict_proba(X_test)

    # print(classification_report(y_true, y_pred))
    results = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return results[0], results[1], results[2], y_pred, scores


def read_image_type_features(file_path):
    image_features = {}

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile)
        # next(readCSV)
        for row in readCSV:
            # print(row)
            doc_id = row[0]
            image_id = row[1]
            panel_id = row[2]
            type_feature = row[3][2:-2]
            type_feature = [float(s) for s in type_feature.split(' ') if len(s) > 1]
            # print(type_feature)
            if doc_id not in image_features.keys():
                image_features[doc_id] = {}
            if image_id not in image_features[doc_id].keys():
                image_features[doc_id][image_id] = {}
            if panel_id not in image_features[doc_id][image_id].keys():
                image_features[doc_id][image_id][panel_id] = type_feature

    for doc in image_features:
        doc_feature = [0] * 12
        for img in image_features[doc]:
            feature = [0] * 12
            for panel in image_features[doc][img]:
                idx = image_features[doc][img][panel].index(max(image_features[doc][img][panel]))
                feature[idx] += 1
                doc_feature[idx] +=1
            image_features[doc][img]['type_feature'] = feature
        image_features[doc]['type_feature'] = doc_feature



    return image_features




def show_statistics_on_data(train_data,image_types):
    # panel type distribution in number
    relevant_panel_hist = np.zeros(12)
    irrelevant_panel_hist = np.zeros(12)
    # panel type distribution binary
    relevant_panel_hist_b = np.zeros(12)
    irrelevant_panel_hist_b = np.zeros(12)
    # number of images per doc
    relevant_image_list = []
    irrelevant_image_list = []
    # panel number per images
    relevant_panel_list = []
    irrelevant_panel_list = []

    for doc in train_data['yes']:
        relevant_panel_hist += np.array(image_types[doc]['type_feature'])
        relevant_panel_hist_b += (np.array(image_types[doc]['type_feature'])>0).astype(int)
        relevant_image_list.append(len(image_types[doc].keys()) - 1)
        for img in image_types[doc]:
            if img != 'type_feature':
                relevant_panel_list.append(len(image_types[doc][img]))
                # for panel in image_types[doc][img]:
                #     panel_path = os.path.join('/media/pengyuan/Research/DOC/JAX/output', doc, 'panels', img, panel+'.jpg')
                #     idx = image_types[doc][img][panel].index(max(image_types[doc][img][panel]))
                #     dst_path = os.path.join('/media/pengyuan/Research/DOC/JAX/check_classification_CLEF/all', str(idx),
                #                             doc+ '_' + img+ '_' +panel + '.jpg')
                #     copyfile(panel_path, dst_path)


    for doc in train_data['no']:
        irrelevant_panel_hist += np.array(image_types[doc]['type_feature'])
        irrelevant_panel_hist_b += (np.array(image_types[doc]['type_feature']) > 0).astype(int)
        irrelevant_image_list.append(len(image_types[doc].keys()) - 1)
        for img in image_types[doc]:
            if img != 'type_feature':
                irrelevant_panel_list.append(len(image_types[doc][img]))
                # for panel in image_types[doc][img]:
                #     panel_path = os.path.join('/media/pengyuan/Research/DOC/JAX/output', doc, 'panels', img, panel+'.jpg')
                #     idx = image_types[doc][img][panel].index(max(image_types[doc][img][panel]))
                #     dst_path = os.path.join('/media/pengyuan/Research/DOC/JAX/check_classification_CLEF/all', str(idx),
                #                             doc+ '_' +img + '_' + panel + '.jpg')
                #     copyfile(panel_path, dst_path)


    print('Relevant panel type distribution')
    print(relevant_panel_hist)
    print('Irrelevant panel type distribution')
    print(irrelevant_panel_hist)
    print('Relevant panel type distribution binary')
    print(relevant_panel_hist_b)
    print('Irrelevant panel type distribution binary')
    print(irrelevant_panel_hist_b)

    # data = [relevant_image_list, irrelevant_image_list]
    # plt.boxplot(data, notch=True, patch_artist=True)
    # plt.show()
    # plt.close()
    #
    # data = [relevant_panel_list, irrelevant_panel_list]
    # plt.boxplot(data, notch=True, patch_artist=True)
    # plt.show()
    # plt.close()

    # sum(relevant_image_list)/len(relevant_image_list)
    # median(relevant_image_list)
    # sum(irrelevant_image_list) / len(irrelevant_image_list)
    # median(irrelevant_image_list)
    #
    # sum(relevant_panel_list)/len(relevant_panel_list)
    # median(relevant_panel_list)
    # sum(irrelevant_panel_list) / len(irrelevant_panel_list)
    # median(irrelevant_panel_list)

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

def create_image_type_feature(img):
    feature = [0]*12
    for panel in img:
        idx = img[panel]['type']
        feature[idx] = feature[idx] +1

def show_statistics(text_data):

    image_type_comb_all = []
    image_type_comb_yes = []
    image_type_comb_no = []
    for doc in text_data['yes']:
        if 'imgs' in text_data['yes'][doc].keys():
            for img in text_data['yes'][doc]['imgs']:
                image_type_comb_all.append(text_data['yes'][doc]['imgs'][img]['type_pattern'])
                image_type_comb_yes.append(text_data['yes'][doc]['imgs'][img]['type_pattern'])

    for doc in text_data['no']:
        if 'imgs' in text_data['no'][doc].keys():
            for img in text_data['no'][doc]['imgs']:
                image_type_comb_all.append(text_data['no'][doc]['imgs'][img]['type_pattern'])
                image_type_comb_no.append(text_data['no'][doc]['imgs'][img]['type_pattern'])

    hist_all = [convert_binary_to_int(type_comb) for type_comb in image_type_comb_all]
    hist_yes = [convert_binary_to_int(type_comb) for type_comb in image_type_comb_yes]
    hist_no = [convert_binary_to_int(type_comb) for type_comb in image_type_comb_no]
    counts_all = Counter(hist_all)
    counts_yes = Counter(hist_yes)
    counts_no = Counter(hist_no)
    counts_all = counts_all.items()
    counts_yes = counts_yes.items()
    counts_no = counts_no.items()
    counts_yes = sorted(counts_yes, key=lambda x: x[1], reverse=True)
    counts_no = sorted(counts_no, key=lambda x: x[1], reverse=True)
    counts_all = sorted(counts_all, key=lambda x: x[1], reverse=True)
    feature_list = [item[0] for item in counts_all if item[1]>5]
    print(counts_yes)
    print(counts_no)
    return feature_list

def add_pattern_feature(data, feature_list):

    for doc in data['yes']:
        if 'imgs' in data['yes'][doc].keys():
            data['yes'][doc]['vectors']['IMG_pattern'] = create_pattern_feature(data['yes'][doc], feature_list)

    for doc in data['no']:
        if 'imgs' in data['no'][doc].keys():
            data['no'][doc]['vectors']['IMG_pattern'] = create_pattern_feature(data['no'][doc], feature_list)


    return data

def convert_binary_to_int(image_type_feature):
    binary_feature = ''
    for type in image_type_feature:
        if type>0:
            binary_feature = binary_feature + '1'
        else:
            binary_feature = binary_feature + '0'
    return int(binary_feature, 2)

def show_features(feature_list):
    type_list = ['3D', 'BarChart', 'Fluorence', 'Gel', 'LineChart',
                 'LightMicroscopy', 'OtherGraph', 'Plate', 'Sequence',
                 'Text', 'Unknown', 'Picture']
    count = 0
    for feature in feature_list:
        b_feature = f'{feature:012b}'
        output = []
        for type_idx in range(len(b_feature)):
            if b_feature[type_idx] == '1':
                output.append(type_list[type_idx])
        print('Order ' + str(count))
        print(feature)
        print(output)
        count += 1

def main():
    # load text data file

    with open('GXD_data_vectors_classifiers_0715.json') as f:
        text_data = json.load(f)
    # with open('results_comparision_12_1220.json') as f:
    #     results = json.load(f)


    feature_list = show_statistics(text_data)
    # data = add_pattern_feature(text_data, feature_list)
    data_classifiers = text_data

    show_features(feature_list)


    for feature_no in np.arange(10, 91, 10):
        precision = []
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
                if 'IMG_type' in text_data['yes'][doc].keys():
                    # print(doc)
                    if idx % 5 != fold_idx:
                        train_data['yes'][doc] = text_data['yes'][doc]
                    if idx % 5 == fold_idx:
                        test_data['yes'][doc] = text_data['yes'][doc]
                        test_list.append(doc)

                    idx += 1
                # else:
                #
                #     #print(doc)

            train_data['no'] = {}
            test_data['no'] = {}
            idx = 0
            for doc in text_data['no']:
                if 'IMG_type' in text_data['no'][doc].keys():
                    if idx % 5 != fold_idx:
                        train_data['no'][doc] = text_data['no'][doc]
                    if idx % 5 == fold_idx:
                        test_data['no'][doc] = text_data['no'][doc]
                        test_list.append(doc)

                    idx += 1
                # else:
                #     #print(doc)

            # show_statistics_on_data(train_data,image_types)
            # show_statistics_on_data(test_data, image_types)

            # vector_generator = baseline_vector_generator(train_data, feature_no) # vector_generator_caption
            train_X, train_Y, test_X, test_Y = feature_creation(train_data, test_data, feature_no, feature_list)

            p, r, f, predictions, scores = svm_classification(train_X, train_Y, test_X, test_Y)
            precision.append(p)
            recall.append(r)
            fscore.append(f)

            tp = [j for i, j in zip(test_Y, predictions) if i == 1 and j == 1]
            fp = [j for i, j in zip(test_Y, predictions) if i == 0 and j == 1]
            u10 = (10 * sum(tp) - sum(fp)) / (10 * (sum(tp) + sum(fp)))
            u20 = (20 * sum(tp) - sum(fp)) / (20 * (sum(tp) + sum(fp)))
            utility10.append(u10)
            utility20.append(u20)

            results = update_classification_result(data_classifiers, test_list, predictions, scores, 'IMG_pattern')

        print('Results on ' + str(feature_no))
        print(precision)
        print(recall)
        print(fscore)
        print(utility10)
        print(utility20)
        print('precsion: ', sum(precision) / len(precision), statistics.stdev(precision))
        print('recall: ', sum(recall) / len(recall), statistics.stdev(recall))
        print('fscore: ', sum(fscore) / len(fscore), statistics.stdev(fscore))
        print('-' * 20)

    with open('GXD_data_vectors_classifiers_0715.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()