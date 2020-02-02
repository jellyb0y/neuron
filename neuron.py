from math import exp
from random import random


class Exceptions():
        '''
            Класс ошибок
        '''
        InvalidValue = Exception('Invalid value')
        InvalidDim = Exception('Invalid dimention of vector')


class Neuron():
    '''
        Класс нейрона
    '''

    def __init__(self, inputs_count, a_coef):
        if not isinstance(inputs_count, int) or not isinstance(a_coef, float):
            raise ValueError

        if inputs_count < 1:
            raise Exceptions.InvalidValue

        self.inputs_count = inputs_count
        self.a_coef = a_coef
        self.weights = self.gen_weights(self.inputs_count + 1)

    def gen_weights(self, inputs_count):
        array = []
        for _ in range(inputs_count):
            array.append(random())

        return array

    def func(self, sum_):
        '''
            Метод вычисления эксп. функции
        '''
        return 1 / (1 + exp(-self.a_coef * sum_))   

    def sum(self, x, weights):
        '''
            Сумматор
        '''
        sum_ = 1 * weights[0]
        for i in range(len(x)):
            sum_ += x[i] * weights[i + 1]

        return sum_

    def calc(self, x, weights=None):
        '''
            Внешний открытый метод вычисления выходного сигнала
        '''
        if not weights is None:
            if not isinstance(weights, list):
                raise ValueError
            if len(self.weights) != self.inputs_count + 1:
                raise Exceptions.InvalidDim
            self.weights = weights

        if not isinstance(x, list):
            raise ValueError

        if len(x) != self.inputs_count:
            raise Exceptions.InvalidDim

        sum_ = self.sum(x, self.weights)
        result = self.func(sum_)

        return result

    def learn(self, a, learning_constant, error):
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[0] += error * learning_constant
            else: 
                self.weights[i] += error * learning_constant * a[i - 1]

        return self.weights

    def sum_miss(self, a, expected_arr):
        sum_ = 0
        for i in range(len(a)):
            result = self.calc(a[i], self.weights)
            miss = (expected_arr[i] - result)**2
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
            if len(vec) != self.inputs_count:
                raise Exceptions.InvalidDim

        error = None
        while (gen is None and error != 0) or (not gen is None and gen > 0):
            for i in range(len(a)):
                error_ = expected_arr[i] - self.calc(a[i])
                self.weights = self.learn(a[i], learning_constant, error_)
            if gen is None or log:
                error = self.sum_miss(a, expected_arr)
            else:
                gen -= 1
            if log:
                error = self.sum_miss(a, expected_arr)
                print(error, self.weights)
        return self.weights


class NeuronDigit(Neuron):
    '''
        Класс нейрона
        Возвращает 1, 0 или -1
    '''

    def func(self, sum_):
        '''
            Метод вычисления эксп. функции
        '''
        sigmoid = Neuron.func(self, sum_)

        if sigmoid > 0.75:
            return 1
        elif sigmoid <= 0.75 and sigmoid >= 0.25:
            return 0
        else: return -1