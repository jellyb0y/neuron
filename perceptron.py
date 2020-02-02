from neuron import NeuronDigit, Exceptions

class Perceptron():
    '''
        Класс перцептрона
    '''

    def __init__(self, detectors_dim, neurons_dim, neurons_a, layers_count, reacts_dim, reacts_a):
        if not isinstance(detectors_dim, int) \
        or not isinstance(neurons_dim, int) \
        or not isinstance(layers_count, int) \
        or not isinstance(reacts_dim, int) \
        or not isinstance(neurons_a, float) \
        or not isinstance(reacts_a, float):
            raise ValueError

        self.detectors_dim = detectors_dim

        if detectors_dim < 1 \
        or neurons_dim < 1 \
        or layers_count < 1 \
        or reacts_dim < 1:
            raise Exceptions.InvalidValue

        self.reacts = self.gen_neuron_layer(reacts_dim, neurons_dim, reacts_a)
        self.layers = self.gen_layers(neurons_dim, detectors_dim, neurons_a, layers_count)

    def gen_neuron_layer(self, dim, inputs, a):
        layer = []
        for _ in range(dim):
            layer.append(Neuron(inputs, a))

        return layer

    def set_weights(self, weights):
        if not isinstance(weights, list):
            raise ValueError

        for i in range(len(self.layers)):
            layer = self.layers[i]
            for j in range(len(layer)):
                layer[j].weights = weights[i][j]

        for i in range(len(self.reacts)):
            self.reacts[i].weights = weights[len(weights) - 1][i] 

    def give_weights(self):
        weights = []

        for layer in self.layers:
            layer_weight = []
            for neuron in layer:
                layer_weight.append(neuron.weights)
            weights.append(layer_weight)

        reacts_weight = []
        for react in self.reacts:
            reacts_weight.append(react.weights)
        weights.append(reacts_weight)

        return weights

    def gen_layers(self, dim, inputs, a, count_layers):
        layers = []
        for i in range(count_layers):
            if i == 0:
                layer = self.gen_neuron_layer(dim, inputs, a)
            else:
                layer = self.gen_neuron_layer(dim, dim, a)
            layers.append(layer)

        return layers

    def calc_x(self, layer, x):
        result = []
        for neuron in layer:
            result.append(neuron.calc(x))

        return result

    def calc(self, x):
        if not isinstance(x, list):
            raise ValueError

        if len(x) != self.detectors_dim:
            raise Exceptions.InvalidDim

        layers = self.layers
        reacts = self.reacts
        self.x_arr = [x, ]

        for layer in self.layers:
            x = self.calc_x(layer, x)
            self.x_arr.append(x)

        result = self.calc_x(reacts, x)

        return result

    def sum_miss(self, a, expected_arr):
        sum_ = 0
        for j in range(len(expected_arr)):
            result = self.calc(a[j])
            for i in range(len(result)):
                miss = (expected_arr[j][i] - result[i])**2
                sum_ += miss
     
        return sum_

    def learning(self, a, expected_arr, learning_constant, gen=None, log=False):
        if not gen is None:
            if not isinstance(gen, int):
                raise ValueError
            if gen < 1:
                raise Exceptions.InvalidValue

        if not isinstance(a, list) or not isinstance(expected_arr, list):
            raise ValueError

        for vec in a:
            if len(vec) != self.detectors_dim:
                raise Exceptions.InvalidDim

        for vec in expected_arr:
            if len(vec) != len(self.reacts):
                raise Exceptions.InvalidDim

        error = None
        while (gen is None and error != 0) or (not gen is None and gen > 0):
            for i in range(len(a)):
                self.learn_perceptron(a[i], expected_arr[i], learning_constant)
            if gen is None:
                error = self.sum_miss(a, expected_arr)
            else:
                gen -= 1
            if log:
                error = self.sum_miss(a, expected_arr)
                print(error)
        return error


    def learn_perceptron(self, a, expected, learning_constant):
        results = self.calc(a)
        reacts = self.reacts
        layers = self.layers
        x_arr = self.x_arr

        errors = self.learn_reacts(reacts, x_arr[len(x_arr) - 1], results, expected, learning_constant)
        self.learn_layers(layers, x_arr, errors, learning_constant)

    def learn_layer(self, layer, x, errors, learning_constant):
        errors = []
        for i in range(len(layer)):
            error = sum(errors)
            layer[i].learn(x, learning_constant, error)
            errors.append(error / len(x))

        return errors

    def learn_layers(self, layers, x_arr, errors, learning_constant):
        for i in range(len(layers) - 1, -1, -1):
            errors = self.learn_layer(layers[i], x_arr[i], errors, learning_constant)

    def learn_reacts(self, layer, x, results, ecpected_arr, learning_constant):
        errors = []
        for i in range(len(layer)):
            error = ecpected_arr[i] - layer[i].calc(x)
            layer[i].learn(x, learning_constant, error)
            errors.append(error / len(x))

        return errors


class PerceptronDigit(Perceptron):
    '''
        Класс цифрового персептрона
    '''

    def gen_neuron_layer(self, dim, inputs, a):
        layer = []
        for _ in range(dim):
            layer.append(NeuronDigit(inputs, a))

        return layer
