import copy
import os
import pickle
import random
from cgi import test
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ahocorapy.keywordtree import KeywordTree
from fastparquet import ParquetFile
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import utilities
from predict_attributes import AttributePredictor


def create_search_tree(attr_idx_label):
    """Build a search tree."""

    new_labels_idx = dict()  # Maps from the labels of each class to the index of the class.
    kwtree = KeywordTree(case_insensitive=True)  # We don't care about the case.
    # To avoid recognizing a word that is only a subword. But still keep plural, ...
    label_suffix = [".", "!", "?", ":", " ", ",", ";", "-",
                    "s.", "s!", "s?", "s:", "s ", "s,", "s;", "s-",
                    "es.", "es!", "es?", "es:", "es ", "es,", "es;", "es-"]
    # for attr_idx in attr_idx_attr_list_dict.keys():
    for attr_idx in attr_idx_label.keys():  # Different versions/declinations of the same word.
        label_list = attr_idx_label[attr_idx]
        if type(attr_idx_label[attr_idx]) != list:  # Make it a list.
            label_list = [attr_idx_label[attr_idx]]

        for label in label_list:
            for sfx in label_suffix:
                label1 = " " + label + sfx
                kwtree.add(label1)  # Doesn't work with regex.
                if label1 not in new_labels_idx.keys():
                    new_labels_idx[label1] = attr_idx
                else:
                    print("Error, should not happen")
                    # new_labels_idx[label1].append(attr_idx)
    kwtree.finalize()
    return kwtree, new_labels_idx


def get_names(adder=""):
    """Return all the names of the Laion400M text files."""

    # Search in all text data of LAION400M after the labels:
    names_in = []
    names_out = []
    for i in range(0, 32):
        if i < 10:
            numb = "0" + str(i)
        else:  # For 2 digit numbers.
            numb = str(i)
        name_in = "/data/felix/Laion400M/part-000" + numb + "-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
        name_out = "/data/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/" + adder + "result_" + numb + ".pickle"
        names_in.append(name_in)
        names_out.append(name_out)

    return names_in, names_out


def count_sampels(names_in, names_out, kwtree, new_labels_idx):
    """Count the samples."""

    for name_idx in range(len(names_in)):
        name_in = names_in[name_idx]
        name_out = names_out[name_idx]

        pf = ParquetFile(name_in)
        dff = pf.to_pandas()["TEXT"]
        cleared_dff = dff.dropna()  # Samples with Nan as text should not be counted.

        # Calculate occurences per file:
        result_mtx = np.zeros((len(cleared_dff), 86), dtype=bool)  # list()
        cnt = 0
        for index, value in cleared_dff.items():  # [:int(len(dff) / 10)]
            # Select important text from the line.
            text = " " + value + " "
            # To have a whitespace at the beginning and at the end.
            # Check class occurances in the text:
            result = kwtree.search_all(text)
            for found in result:
                # print("Count: ", cnt, "--------", found)
                keyword = found[0]
                pos = new_labels_idx[keyword]
                # Don't count occurences. If there are several same occurences, still 1 should be saved.
                result_mtx[cnt][pos] = True
                # print(pos, row[pos])
            cnt += 1
        with open(name_out, "wb") as act:
            pickle.dump(result_mtx, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        print("Done file number: ", name_idx)


def get_occurrence(names_out, idx_label):
    """Save the last word occurrences of each class."""

    # Calculate relative occurence of the classes in the english training samples.
    samples = 0
    occurence = np.zeros(len(idx_label.keys()) + 1)
    a = np.shape(occurence)
    for name_out in names_out:
        with open(name_out, "rb") as act:
            result_mtx = pickle.load(act)
            b = np.shape(result_mtx)
        act.close()
        samples += np.shape(result_mtx)[0]
        # print(samples)
        # print(len(np.sum(result_mtx, axis=0)))
        occurence += np.sum(result_mtx, axis=0)
        print("One step done.")
    # occurence = occurence / samples

    print("Samples: ", samples)
    print("Occurences: ", occurence)
    return occurence


def main():
    attrPred = AttributePredictor()
    kwtree, new_labels_idx = create_search_tree(attrPred.idx_attribute)

    with open("/data/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/aho_corasick_class_repr_only_ending.pickle", "wb") as act:
        pickle.dump(kwtree, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/data/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_labels_idx_only_ending.pickle", "wb") as act:
        pickle.dump(new_labels_idx, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()

    names_in, names_out = get_names()
    count_sampels(names_in, names_out, kwtree, new_labels_idx)
    occurrence = get_occurrence(names_out, attrPred.idx_attribute)
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/absolute_occurence.pickle", "wb") as act:
        pickle.dump(occurrence, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def save_attr_occ_mapping():
    """Save a mapping from the attribute index or name to it's nubmer of occurrences."""

    attrPred = AttributePredictor()
    idx_attribute = attrPred.idx_attribute

    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/absolute_occurence.pickle", "rb") as act:
        occurrence = pickle.load(act)
    act.close()
    res_mapping = {attr_idx: 0 for attr_idx in idx_attribute.keys()}
    res_mapping_readable = {attr: 0 for attr in idx_attribute.values()}
    for i in idx_attribute.keys():
        # Because idx_attribtue keys starts with 1, but occurrence at index 0 chould be 0 (no entries).
        occ = occurrence[i]
        res_mapping[i] = occ
        res_mapping_readable[idx_attribute[i]] = occ

    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/attr_idx_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/attr_readable_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping_readable, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def main1():
    """Also incorporate that the words can exist out of two words that can be separated by a whitespace."""

    attrPred = AttributePredictor()
    # Added alternative spellings and removed plurals (they are already added later).
    attrPred_new = {1: 'black', 2: 'white', 3: 'blue', 4: 'brown', 5: 'gray', 6: 'orange', 7: 'red', 8: 'yellow', 9: 'patch',
                    10: 'spot', 11: 'stripe', 12: 'furry', 13: "hairless", 14: ["toughskin", "tough skin"], 15: 'big',
                    16: 'small', 17: 'bulbous', 18: 'lean', 19: 'flipper', 20: 'hand', 21: 'hoove', 22: 'pad', 23: 'paw', 24: ['longleg', "long leg"],
                    25: ['longneck', "long neck"], 26: 'tail', 27: ['chewteeth', "chew teeth"], 28: ['meatteeth', "meat teeth"], 29: ['buckteeth', "buck teeth"],
                    30: ['strainteeth', "strain teeth"], 31: 'horn', 32: 'claw', 33: 'tusk', 34: 'smelly', 35: 'fly', 36: 'hop', 37: 'swim', 38: 'tunnel',
                    39: 'walk', 40: 'fast', 41: 'slow', 42: 'strong', 43: 'weak', 44: 'muscle', 45: 'bipedal', 46: ['quadrapedal', "quadrupedal"], 47: 'active',
                    48: 'inactive', 49: 'nocturnal', 50: 'hibernate', 51: 'agility', 52: 'fish', 53: 'meat', 54: 'plankton', 55: 'vegetation', 56: 'insect', 57:
                    'forager', 58: 'grazer', 59: 'hunter', 60: 'scavenger', 61: 'skimmer', 62: 'stalker', 63: ['newworld', "new world"], 64: ['oldworld', "old world"],
                    65: 'arctic', 66: 'coastal', 67: 'desert', 68: 'bush', 69: 'plain', 70: 'forest', 71: 'field', 72: 'jungle', 73: 'mountain', 74: 'ocean',
                    75: 'ground', 76: 'water', 77: 'tree', 78: 'cave', 79: 'fierce', 80: 'timid', 81: 'smart', 82: 'group', 83: 'solitary',
                    84: ['nestspot', "nest spot"], 85: 'domestic'}
    kwtree, new_labels_idx = create_search_tree(attrPred_new)

    with open("/data/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_aho_corasick_class_repr_only_ending.pickle", "wb") as act:
        pickle.dump(kwtree, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/data/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_new_labels_idx_only_ending.pickle", "wb") as act:
        pickle.dump(new_labels_idx, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()

    names_in, names_out = get_names(adder="new_")
    print("Start counting samples")
    count_sampels(names_in, names_out, kwtree, new_labels_idx)
    occurrence = get_occurrence(names_out, attrPred_new)
    print("Counted samples!")
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_absolute_occurence.pickle", "wb") as act:
        pickle.dump(occurrence, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def save_new_attr_occ_mapping():
    """Save a mapping from the attribute index or name to it's nubmer of occurrences."""

    attrPred = AttributePredictor()
    idx_attribute = attrPred.idx_attribute

    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_absolute_occurence.pickle", "rb") as act:
        occurrence = pickle.load(act)
    act.close()
    res_mapping = {attr_idx: 0 for attr_idx in idx_attribute.keys()}
    res_mapping_readable = {attr: 0 for attr in idx_attribute.values()}
    for i in idx_attribute.keys():
        # Because idx_attribtue keys starts with 1, but occurrence at index 0 chould be 0 (no entries).
        occ = occurrence[i]
        res_mapping[i] = occ
        res_mapping_readable[idx_attribute[i]] = occ

    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_idx_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_readable_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping_readable, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


if __name__ == "__main__":
    # main()
    # save_attr_occ_mapping()
    save_new_attr_occ_mapping()
