from abc import abstractmethod
from time import time

import numpy as np

def classify(possibilities):
    return np.rint(possibilities[:, -1])

class Activation:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, inputs):
        pass

    @abstractmethod
    def derivative(self, inputs):
        pass

class ActivationReLU(Activation):
    def __call__(self, inputs, normalized=True):
        output = np.maximum(0, inputs)
        if normalized:
            max_num = np.max(output, axis=1, keepdims=True)
            scale_rate = np.where(max_num == 0, 1, 1 / max_num)
            output = output * scale_rate
        return output

    def derivative(self, inputs):
        return np.where(inputs > 0, 1, 0)

class ActivationSigmoid(Activation):
    def __call__(self, inputs):
        output = 1 / (1 + np.exp(-inputs))  # 正确的 Sigmoid 函数公式
        return output

    def derivative(self, inputs):
        output = self.__call__(inputs) #  关键：使用 __call__ 方法获取 Sigmoid 输出
        return output * (1 - output)   # 正确的 Sigmoid 函数导数公式


class ActivationSoftmax(Activation):
    def __call__(self, inputs):
        slided_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_inputs = np.exp(slided_inputs)
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    def derivative(self, inputs):
        raise NotImplementedError("Softmax derivative is typically handled within the loss function's backpropagation.")

class Loss:
    @abstractmethod
    def __call__(self, predicted, real):
        pass

    @abstractmethod
    def derivative(self, predicted, real, activation: Activation):
        pass

class CrossEntropyLoss(Loss):
    def __call__(self, predicted, real):
        predicted_clipped = np.clip(predicted, 1e-7, 1 - 1e-7)
        loss = -np.sum(np.log(predicted_clipped) * real) / real.shape[0]
        return loss

    def derivative(self, predicted, real, activation: Activation):
        if type(activation) == ActivationSoftmax:
            return predicted - real
        elif type(activation) == ActivationSigmoid:
            return predicted - real
        else:
            raise Exception("Unsupported activation.")

class Layer:
    def __init__(self, n_inputs, n_neurons, activation: Activation):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs) # He initialization
        self.biases = np.zeros(n_neurons) # Initialize biases to zero
        self.activation = activation
        self.linear_output = None
        self.inputs = None

        # Optimizer states (for Momentum, RMSprop, Adam)
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)
        self.weight_velocity = np.zeros_like(self.weights) # For RMSprop and Adam
        self.bias_velocity = np.zeros_like(self.biases)     # For RMSprop and Adam
        self.weight_adam_m = np.zeros_like(self.weights) # Adam: first moment vector
        self.bias_adam_m = np.zeros_like(self.biases)     # Adam: first moment vector
        self.weight_adam_v = np.zeros_like(self.weights) # Adam: second moment vector
        self.bias_adam_v = np.zeros_like(self.biases)     # Adam: second moment vector


    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.biases # Added biases to linear output
        outputs = self.activation(self.linear_output)
        return outputs

    def backward(self, post_diff, post_weights, learning_rate, optimizer_type='SGD', momentum_rate=0.9,
                 rms_prop_decay_rate=0.999, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7, last_layer=False,
                 epoch=None):
        # 如果是最后一层则误差项直接提供
        if not last_layer:
            diff = np.dot(post_diff, post_weights.T) * self.activation.derivative(self.linear_output)
        else:
            diff = post_diff

        batch_size = self.inputs.shape[0]
        gradient_w = np.dot(self.inputs.T, diff) / batch_size
        gradient_b = np.sum(diff, axis=0) / batch_size  # Removed keepdims=True
        # gradient_b = np.sum(diff, axis=0, keepdims=True) / batch_size # Original line with keepdims=True

        # Weight updates based on optimizer type
        if optimizer_type == 'SGD':
            weight_update = gradient_w * learning_rate
            bias_update = gradient_b * learning_rate

        elif optimizer_type == 'Momentum':
            self.weight_momentum = momentum_rate * self.weight_momentum + gradient_w
            self.bias_momentum = momentum_rate * self.bias_momentum + gradient_b
            weight_update = self.weight_momentum * learning_rate
            bias_update = self.bias_momentum * learning_rate

        elif optimizer_type == 'RMSprop':
            self.weight_velocity = rms_prop_decay_rate * self.weight_velocity + (1 - rms_prop_decay_rate) * np.square(
                gradient_w)
            self.bias_velocity = rms_prop_decay_rate * self.bias_velocity + (1 - rms_prop_decay_rate) * np.square(
                gradient_b)
            weight_update = learning_rate * gradient_w / (np.sqrt(self.weight_velocity) + adam_epsilon)
            bias_update = learning_rate * gradient_b / (np.sqrt(self.bias_velocity) + adam_epsilon)

        elif optimizer_type == 'Adam':
            t = epoch + 1  # Adam uses timestep t, epoch+1 to start from t=1
            self.weight_adam_m = adam_beta1 * self.weight_adam_m + (1 - adam_beta1) * gradient_w
            self.bias_adam_m = adam_beta1 * self.bias_adam_m + (1 - adam_beta1) * gradient_b
            self.weight_adam_v = adam_beta2 * self.weight_adam_v + (1 - adam_beta2) * np.square(gradient_w)
            self.bias_adam_v = adam_beta2 * self.bias_adam_v + (1 - adam_beta2) * np.square(gradient_b)

            weight_m_corrected = self.weight_adam_m / (1 - adam_beta1 ** t)
            bias_m_corrected = self.bias_adam_m / (1 - adam_beta1 ** t)
            weight_v_corrected = self.weight_adam_v / (1 - adam_beta2 ** t)
            bias_v_corrected = self.bias_adam_v / (1 - adam_beta2 ** t)

            weight_update = learning_rate * weight_m_corrected / (np.sqrt(weight_v_corrected) + adam_epsilon)
            bias_update = learning_rate * bias_m_corrected / (np.sqrt(bias_v_corrected) + adam_epsilon)


        else:  # Default to SGD if optimizer_type is not recognized
            weight_update = gradient_w * learning_rate
            bias_update = gradient_b * learning_rate

        self.weights -= weight_update
        self.biases -= bias_update

        return diff, self.weights

class NetWork:
    def __init__(self, network_shape, loss=CrossEntropyLoss(), optimizer_type='SGD', optimizer_config=None) -> None: # Added optimizer_type and optimizer_config
        self.shape = network_shape
        self.layers = []
        self.optimizer_type = optimizer_type # Store optimizer type
        self.optimizer_config = optimizer_config if optimizer_config else {} # Store optimizer config, use empty dict if None

        for i in range(len(network_shape) -1):
            if i == len(network_shape) - 2:
                layer = Layer(network_shape[i], network_shape[i+1], ActivationSoftmax())
            else:
                layer = Layer(network_shape[i], network_shape[i+1], ActivationReLU())
            self.layers.append(layer)
        self.loss_func = loss


    def train(self, training_data, targets, validation_data=None, validation_targets=None, batch_size=32, epochs=10, learning_rate=0.01):
        print('Training starts')
        print('-' * 25)
        num_epochs = epochs
        num_samples = training_data.shape[0]
        num_batches = num_samples // batch_size # 计算批次数量，向下取整

        batch_outputs = None
        for epoch in range(num_epochs):
            start_time = time()
            print(f'Epoch {epoch+1}/{num_epochs}')

            # Shuffle training data and targets at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            training_data_shuffled = training_data[permutation]
            targets_shuffled = targets[permutation]

            epoch_train_loss = 0.0
            epoch_train_accuracy = 0.0

            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = (batch_index + 1) * batch_size
                batch_data = training_data_shuffled[start_index:end_index]
                batch_targets = targets_shuffled[start_index:end_index]

                # Forward pass for the batch
                batch_outputs = self.forward(batch_data)

                # Backward pass and parameter update for the batch
                self.backward(batch_outputs, batch_targets, learning_rate=learning_rate, optimizer_type=self.optimizer_type, epoch=epoch, **self.optimizer_config) # Pass optimizer type and config and epoch

                # Calculate loss and accuracy for the batch
                batch_loss = self.loss_func(batch_outputs[-1], batch_targets)
                batch_accuracy = self.accuracy(batch_outputs[-1], batch_targets)

                epoch_train_loss += batch_loss
                epoch_train_accuracy += batch_accuracy

            # Calculate average loss and accuracy for the epoch
            avg_train_loss = epoch_train_loss / num_batches
            avg_train_accuracy = epoch_train_accuracy / num_batches

            print(f'Train loss: {avg_train_loss:.4f}') # 格式化输出 loss
            print(f'Train accuracy: {avg_train_accuracy:.4f}') # 格式化输出 accuracy

            if validation_data is not None and validation_targets is not None:
                validation_outputs = self.forward(validation_data)
                validation_loss = self.loss_func(validation_outputs[-1], validation_targets)
                validation_accuracy = self.accuracy(validation_outputs[-1], validation_targets)
                print(f'Validation loss: {validation_loss:.4f}') # 格式化输出 validation loss
                print(f'Validation accuracy: {validation_accuracy:.4f}') # 格式化输出 validation accuracy

            end_time = time()
            epoch_time = end_time - start_time
            print(f'Epoch time: {epoch_time:.2f} seconds') # 格式化输出时间
            print('-' * 25)

        print('Training complete')
        return batch_outputs[-1]

    @staticmethod
    def accuracy(outputs, targets):
        """
        计算多分类任务的准确率 (例如 MNIST).

        Args:
            outputs (np.ndarray): 神经网络的输出，形状为 (batch_size, num_classes).
            targets (np.ndarray): One-hot 编码的真实标签，形状为 (batch_size, num_classes).

        Returns:
            float: 准确率 (0.0 到 1.0 之间).
        """
        predicted_classes = np.argmax(outputs, axis=1)
        correct_classes = np.argmax(targets, axis=1)
        correct_predictions = np.sum(predicted_classes == correct_classes) # 统计预测正确的数量
        accuracy = correct_predictions / targets.shape[0] # 计算准确率
        return accuracy


    def forward(self, inputs):
        outputs = [np.copy(inputs)]
        layer_output = outputs[0]
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
            outputs.append(layer_output)
        return outputs

    def backward(self, outputs, targets, learning_rate, optimizer_type='SGD', epoch=None, **optimizer_config): # Pass optimizer type and epoch and config
        output = outputs[-1]
        last_layer = self.layers[-1]
        diff = self.loss_func.derivative(output, targets, last_layer.activation)
        diff, weights = last_layer.backward(diff, None, learning_rate, optimizer_type=optimizer_type, last_layer=True, epoch=epoch, **optimizer_config) # Pass optimizer type and epoch and config
        for i in reversed(range(len(self.layers) - 1)):
            diff, weights = self.layers[i].backward(diff, weights, learning_rate, optimizer_type=optimizer_type, epoch=epoch, **optimizer_config) # Pass optimizer type and epoch and config

def main():
    pass


if __name__ == "__main__":
    main()
