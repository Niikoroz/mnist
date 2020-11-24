# Classification of numbers from 0 to 9 (MNIST)
# ----------------------------------------------------------------------------------------------------------------------
import numpy
import scipy.special


class NN:

    def __init__(self, inp, hid, out, learn):
        self.i = inp
        self.h = hid
        self.o = out
        self.lr = learn

        self.wih = numpy.random.rand(self.h, self.i) - 0.5
        self.who = numpy.random.rand(self.o, self.h) - 0.5

        self.activation = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        input_values = numpy.array(inputs_list, ndmin=2).T
        target_values = numpy.array(targets_list, ndmin=2).T

        hidden_values = numpy.dot(self.wih, input_values)
        hidden_active_values = self.activation(hidden_values)

        output_values = numpy.dot(self.who, hidden_active_values)
        output_active_values = self.activation(output_values)

        output_errors = target_values - output_active_values
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * output_active_values * (1.0 - output_active_values)),
                                        numpy.transpose(hidden_active_values))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_active_values * (1.0 - hidden_active_values)),
                                        numpy.transpose(input_values))

        pass

    def work(self, inputs_list):
        input_values = numpy.array(inputs_list, ndmin=2).T

        hidden_values = numpy.dot(self.wih, input_values)
        hidden_active_values = self.activation(hidden_values)

        output_values = numpy.dot(self.who, hidden_active_values)
        output_active_values = self.activation(output_values)

        return output_active_values


# ----------------------------------------------------------------------------------------------------------------------
# 28 * 28 = 784
input_neurons = 784
hidden_neurons = 100
output_neurons = 10

learning_rate = 0.5

net = NN(input_neurons, hidden_neurons, output_neurons, learning_rate)

train_file = open("mnist_dataset/mnist_train.csv", 'r')
train_list = train_file.readlines()
train_file.close()

epochs = 1
for e in range(epochs):
    for record in train_list:
        all_pixels = record.split(',')
        inputs = (numpy.asfarray(all_pixels[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_neurons) + 0.01
        targets[int(all_pixels[0])] = 0.99
        net.train(inputs, targets)
        pass
pass

test_file = open("mnist_dataset/mnist_test.csv", 'r')
test_list = test_file.readlines()
test_file.close()

right_values = 0
all_values = 0
for record in test_list:
    all_pixels = record.split(',')
    aim = int(all_pixels[0])
    print("Expected value - ", aim)
    inputs = (numpy.asfarray(all_pixels[1:]) / 255.0 * 0.99) + 0.01
    outputs = net.work(inputs)
    label = numpy.argmax(outputs)
    print("Value of net - ", label)
    if label == aim:
        right_values += 1
    all_values += 1
    pass

print("Efficiency of net: ", (right_values / all_values) * 100, "%")
