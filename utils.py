'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

import matplotlib.pyplot as plt


def plot_loss(loss_train, loss_test):
    fig, ax = plt.subplots()
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss (Cost Function)')
    plt.plot(loss_train, 'b-', label='Training loss per generation')
    plt.plot(loss_test, 'b-', label='Testing loss per generation')
    ax.legend(loc='upper right', shadow=True)
    plt.grid()
    plt.show()


def plot_accuracy(accuracy_train, accuracy_test):
    fig, ax = plt.subplots()
    plt.title('Training Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy (%)')
    plt.plot(accuracy_train, 'b-', label='Generation accuracy (training data)')
    plt.plot(accuracy_test, 'b-', label='Generation accuracy (testing data')
    ax.legend(loc='lower right', shadow=True)
    plt.grid()
    plt.show()
