import numpy as np

from minst_data import load_mnist_data
from round_data import create_round_data, plot_data
from network import NetWork


# def main():
#     data = create_round_data(10000)
#     training_data = data[:, :-1]
#     targets = data[:, -1:]
#     # print('training data :', training_data)
#     data = create_round_data(1000)
#     validation_data = data[:, :-1]
#     validation_targets = data[:, -1:]
#     plot_data(data, "data")
#     network = NetWork(
#         [2, 64, 32, 1],
#         # optimizer_type='Adam',
#         # optimizer_config={
#         #     'adam_beta1': 0.9,
#         #     'adam_beta2': 0.999,
#         # }
#
#         # optimizer_type='Momentum',
#         # optimizer_config={
#         #     'momentum_rate': 0.9
#         # }
#
#         optimizer_type='RMSprop',
#         optimizer_config={
#             'rms_prop_decay_rate': 0.999,
#             'adam_epsilon': 1e-7,
#         }
#     )
#     # print('training targets :', targets)
#     network.train(training_data, targets, validation_data, validation_targets, epochs=100, learning_rate=0.02)
#     outputs = network.forward(validation_data)
#
#     print('accuracy: ', network.accuracy(outputs[-1], validation_targets))
#     data[:, -1:] = np.rint(outputs[-1]).astype(int)
#     plot_data(data, "after")

def main():
    training_data, training_labels = load_mnist_data('./dataset/minst/train', './dataset/minst/train_labs.txt')
    validation_data, validation_labels = load_mnist_data('./dataset/minst/test', './dataset/minst/test_labs.txt')

    network = NetWork(
        [784, 256, 128, 10],
        optimizer_type='Adam',
        optimizer_config={
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
        }
    )

    network.train(training_data, training_labels, validation_data, validation_labels, epochs=25)




if __name__ == '__main__':
    main()