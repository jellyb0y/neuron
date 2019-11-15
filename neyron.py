from math import exp
from random import random


class Exceptions():
		'''
			Класс ошибок
		'''
		InvalidInputsCount = Exception('Invalid count of inputs')
		InvalidDim = Exception('Invalid dimention of vector')


class Neyron():
	'''
		Класс нейрона
	'''


	def __init__(self, inputs, a):
		if not isinstance(inputs, int) or not isinstance(a, float):
			raise ValueError

		if inputs < 1:
			raise Exceptions.InvalidInputsCount

		self.inputs = inputs
		self.a = a
		self.w = self.__gen_w(inputs)


	def __gen_w(self, inputs):
		arr = []
		for _ in range(inputs):
			arr.append(random())

		return arr


	def __func(self, sum):
		'''
			Метод вычисления эксп. функции
		'''
		a = self.a
		return 1 / (1 + exp(a * sum))


	def __sum(self, x, w):
		'''
			Сумматор
		'''
		k = 0
		sum_ = 0
		for i in x:
			sum_ += i * w[k]
			k += 1

		return sum_


	def calc(self, x, w=None):
		'''
			Внешний открытый метод вычисления выходного сигнала
		'''
		if w is None:
			w = self.__gen_w(self.inputs)

		if not isinstance(x, list) or not isinstance(w, list):
			raise ValueError

		if len(x) != self.inputs or len(w) != self.inputs:
			raise Exceptions.InvalidDim

		sum_ = self.__sum(x, w)
		result = self.__func(sum_)

		return result
