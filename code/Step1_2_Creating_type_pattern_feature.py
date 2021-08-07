# Deep learning for image classification

# Train from scratch for image classification


import keras
from keras.preprocessing import image
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from keras.applications.vgg16 import VGG16
import json
from collections import Counter


def update_image_class_label(train_data):


    doc_classes = ['yes', 'no']
    for type in doc_classes:
        for doc in train_data[type]:
            train_data[type][doc]['imgs'] = {}


    for doc_class in doc_classes:
        clf_resutls_path = os.path.join('/mnt/Research/DOC/JAX/classification_on_Luis', doc_class)
        types = os.listdir(clf_results_path)
        for type in types:
            panel_path = os.path.join(clf_results_path, type)
            if os.path.isdir(panel_path):
                panels = os.listdir(panel_path)
                for panel in panels:
                    if panel.endswith('.jpg'):
                        tmp = panel.split('_')
                        doc = tmp[0]
                        img = tmp[1]+'_'+tmp[2]
                        p = tmp[3][:-4]

                        if doc in train_data[doc_class]:
                            if img not in train_data[doc_class][doc]['imgs']:
                                train_data[doc_class][doc]['imgs'][img] = {}
                            train_data[doc_class][doc]['imgs'][img][p] = {}
                            train_data[doc_class][doc]['imgs'][img][p]['type'] = int(type)


    return train_data

def add_type_pattern_feature(data):
    for doc_class in ['yes', 'no']:
        for doc in data[doc_class]:
            for img in data[doc_class][doc]['imgs']:
                feature = [0] * 12
                for panel in data[doc_class][doc]['imgs'][img]:
                    idx = data[doc_class][doc]['imgs'][img][panel]['type']
                    feature[idx] = feature[idx] + 1
                data[doc_class][doc]['imgs'][img]['type_pattern'] = feature

    return data

def img_feature_analysis(doc_data):
    print(doc_data['vectors']['IMG_pattern'])
    print([x[0] for x in enumerate(doc_data['vectors']['IMG_pattern']) if x[1] >0])

def convert_binary_to_int(image_type_feature):
    binary_feature = ''
    for type in image_type_feature:
        if type>0:
            binary_feature = binary_feature + '1'
        else:
            binary_feature = binary_feature + '0'
    return int(binary_feature, 2)

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

def create_pattern_feature(doc_features, feature_list):
    vector = [0]*len(feature_list)
    for img in doc_features['imgs']:

        type = convert_binary_to_int(doc_features['imgs'][img]['type_pattern'])
        if type in feature_list:
            vector[feature_list.index(type)] += 1
    # print(vector)
    return vector


def add_pattern_feature(data, feature_list):

    for doc in data['yes']:
        if len(data['yes'][doc]['imgs'].keys()) > 0:
            data['yes'][doc]['vectors']['IMG_pattern'] = create_pattern_feature(data['yes'][doc], feature_list)

    for doc in data['no']:
        if len(data['no'][doc]['imgs'].keys()) > 0:
            data['no'][doc]['vectors']['IMG_pattern'] = create_pattern_feature(data['no'][doc], feature_list)


    return data

def main():

    with open('GXD_data_vectors_classifiers_0615.json') as f:
        data = json.load(f)

    data = update_image_class_label(data)

    with open('GXD_data_vectors_classifiers_0715.json', 'w') as f:
        json.dump(data, f)



    data = add_type_pattern_feature(data)


    feature_list = show_statistics(data)

    data = add_pattern_feature(data, feature_list)

    with open('GXD_data_vectors_classifiers_0715.json', 'w') as f:
        json.dump(data, f)





if __name__ == "__main__":
    main()
