'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

import hyperparams as hp
import numpy as np


def import_data(name, headers=False):

    file_name = hp.data_dir + '\\' + name

    with open(file_name) as fp:
        lines = fp.read().split("\n")

    iterator = iter(lines)
    if headers:
        next(iterator)

    data = [line.split(',') for line in iterator]
    return data


# replace zeros in the data set with mean values
# ! columns are expected to contain features ! #
def zero_to_mean(data):
    data = np.transpose(data)
    means = np.sum(data, axis=1) / np.count_nonzero(data, axis=1)
    for col in range(len(data)):
        data[col][data[col] == 0.] = means[col]
    data = np.transpose(data)
    return data


def expand_categorical_feature(feature):
    result = []
    translation = []
    for category in np.unique(feature):
        translation.append(category)
        result.append([1 if row == category else 0 for row in feature])
    return result, translation


def normalize(data):
    col_max = data.max(axis=0)
    col_min = data.min(axis=0)
    return (data - col_min) / (col_max - col_min)
