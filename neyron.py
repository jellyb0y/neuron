from math import exp
from random import random


class Exceptions():
		'''
			Класс ошибок
		'''
		InvalidInputsCount = Exception('Invalid count of inputs')
		InvalidDim = Exception('Invalid dimention of vector')


class Neuron():
	'''
		Класс нейрона
	'''


	def __init__(self, inputs_count, a_coef):
		if not isinstance(inputs_count, int) or not isinstance(a_coef, float):
			raise ValueError

		if inputs_count < 1:
			raise Exceptions.InvalidInputsCount

		self.inputs_count = inputs_count
		self.a_coef = a_coef
		self.weights = self.gen_weight(self.inputs_count + 1)


	def gen_weight(self, inputs_count):
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