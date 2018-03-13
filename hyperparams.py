'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

# Data
data_dir = 'data'

# training
learning_rate = 0.05
batch_size = 500
epochs = 1000
loss_function = 'l2'

# size of validation set in % in relation to entire dataset
validation_size = 0.4

# structure & activations
nn_structure = [10]
nn_activations = ['relu', 'sigmoid']

# auto balance classes
auto_balance = True
