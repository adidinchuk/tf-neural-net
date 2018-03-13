'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

import hyperparams as hp
import data as d
import numpy as np
import network as nwk

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = d.import_data('promoted.csv', headers=True)

features_numeric = d.zero_to_mean([[float(row[2].strip()) if row[2].strip() else 0.,
                     float(row[3].strip()) if row[3].strip() else 0.,
                     float(row[5].strip()) if row[5].strip() else 0.]
                    for row in data])
# normalize numeric features
features_numeric = np.transpose(d.normalize(features_numeric))

features_string = [[row[6].strip() if row[6].strip() else 'N/A',
                    row[7].strip() if row[7].strip() else 'N/A']
                   for row in data]

# expand the categorical columns into binary categories
row4_expansion, row4_translation = d.expand_categorical_feature(np.transpose(features_string)[0])
row4_expansion = row4_expansion
print('Row 4 expanded into the following categories: ' + str(row4_translation))
row5_expansion, row5_translation = d.expand_categorical_feature(np.transpose(features_string)[1])
row5_expansion = row5_expansion
print('Row 5 expanded into the following categories: ' + str(row5_translation))

# concatenate numerical data with categorical features
features = np.append(features_numeric, row4_expansion, axis=0)
features = np.append(features, row5_expansion, axis=0)
features = np.transpose(features)

# extract target data
target = np.array([[float(row[1])] for row in data])

# split into test and train data sets
train_size = round(len(features) * (1 - hp.validation_size))
train_index = np.random.choice(len(features), size=train_size)
test_index = list(set(range(len(features))) - set(train_index))

# training data
train_inputs = features[train_index]
train_outputs = target[train_index]

# testing data
test_inputs = features[test_index]
test_outputs = target[test_index]


nn = nwk.Network(len(features[0]), len(target[0]), hp.nn_structure, hp.nn_activations)
nn.train(train_inputs, train_outputs, test_inputs, test_outputs, auto_balance=hp.auto_balance, plot=False,
         loss_function=hp.loss_function, batch_size=hp.batch_size, epochs=hp.epochs, learning_rate=hp.learning_rate)
