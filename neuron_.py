from numpy import exp, power
from hashlib import md5
from random import random
from time import time


class Exceptions():
	InvalidValue = Exception('Invalid Value was received')
	InvalidObject = Exception('Invalid Object was received')
	InvalidObjectSame = Exception('The same object id was received')
	InvalidID = Exception('Invalid ID was received')
	FullObject = Exception('Object Neuron is already filled up')
	UnexpectedError = Exception('UnexpectedError')


class Symbol():
	"""docstring for Symbol"""

	def __init__(self, string=''):
		if not isinstance(string, str):
			raise ValueError

		self.string = string
		self.__get_value()
		
	def __get_value(self):
		encode = self.string
		rand_ = random()
		time_ = time()

		string = '{}_{}_{}'.format(encode, rand_, time_)
		self.value = md5(string.encode()).hexdigest()


class EmptyNeuron():
	"""docstring for NeuronModel"""

	def __init__(self):
		self.symbol = Symbol('EmptyNeuron')
		self.id_ = self.symbol.value

		self.outputs = None
		self.value = None

	def set_output(self, output):
		if (not isinstance(output, NeuronModel) \
		and not isinstance(output, NeuronDigitModel) \
		and not isinstance(output, EmptyNeuron)):
				raise ValueError

		if output.id_ == self.id_:
			raise Exceptions.InvalidObjectSame

		if not self.outputs:
			self.outputs = {}

		index = output.id_
		self.outputs[index] = output

	def set_value(self, value):
		if (not isinstance(value, float) \
		and not isinstance(value, int)):
			raise ValueError

		self.value = value
		if self.outputs:
			for index in self.outputs:
				self.outputs[index].set_value(self, self.value)


class NeuronModel():
	"""docstring for NeuronModel"""

	def __init__(self, count_inputs, koef=1.0):
		if not isinstance(count_inputs, int) \
		or (not isinstance(koef, float) and not isinstance(koef, int)):
			raise ValueError

		if count_inputs < 1:
			raise Exceptions.InvalidValue

		self.count_inputs = count_inputs
		self.outputs = None
		self.value = None
		self.inputs = {}
		self.inputs_values = {}
		self.inputs_weights = {}
		self.koef = koef
		self.sum = 0

		self.symbol = Symbol('NeuronModel')
		self.id_ = self.symbol.value

	def set_input(self, input_):
		if len(self.inputs) == self.count_inputs:
			raise Exceptions.FullObject

		if (not isinstance(input_, NeuronModel) \
		and not isinstance(input_, NeuronDigitModel) \
		and not isinstance(input_, EmptyNeuron)):
			raise Exceptions.InvalidObject

		if input_.id_ == self.id_:
			raise Exceptions.InvalidObjectSame

		index = input_.id_
		self.inputs[index] = input_
		self.inputs_weights[index] = random()

	def set_value(self, object_, value):
		if (not isinstance(object_, NeuronModel) \
		and not isinstance(object_, NeuronDigitModel) \
		and not isinstance(object_, EmptyNeuron)) \
		or (not isinstance(value, float) and not isinstance(value, int)):
			raise ValueError

		if object_.id_ == self.id_:
			raise Exceptions.InvalidObjectSame

		try:
			index = object_.id_
			if not index in self.inputs:
				raise Exceptions.InvalidObject
			self.inputs_values[index] = value
		except KeyError:
			raise Exceptions.InvalidID

		if len(self.inputs_values) == self.count_inputs:
			self.calculate_value()

	def set_weight(self, object_, weight):
		if (not isinstance(object_, NeuronModel) \
		and not isinstance(object_, NeuronDigitModel) \
		and not isinstance(object_, EmptyNeuron)) \
		or (not isinstance(weight, float) and not isinstance(weight, int)):
			raise ValueError

		if object_.id_ == self.id_:
			raise Exceptions.InvalidObjectSame

		try:
			index = object_.id_
			if not index in self.inputs:
				raise Exceptions.InvalidObject
			self.inputs_weights[index] = weight
		except KeyError:
			raise Exceptions.InvalidID

	def set_output(self, output):
		if (not isinstance(output, NeuronModel) \
		and not isinstance(output, NeuronDigitModel) \
		and not isinstance(output, EmptyNeuron)):
				raise ValueError

		if output.id_ == self.id_:
			raise Exceptions.InvalidObjectSame

		if not self.outputs:
			self.outputs = {}

		index = output.id_
		self.outputs[index] = output

	def activation_func(self):
		return 1 / (1 + exp(-self.koef * self.sum))

	def df_activation_func(self):
		return self.value * (1 - self.value)

	def calculate_value(self):
		values = self.inputs_values
		weights = self.inputs_weights
		self.sum = 0

		for id_ in self.inputs:
			self.sum += values[id_] * weights[id_]

		self.inputs_values = {}
		self.value = self.activation_func()
		
		if self.outputs:
			for index in self.outputs:
				self.outputs[index].set_value(self, self.value)


class NeuronDigitModel(NeuronModel):
	"""docstring for Symbol"""

	def activation_func(self):
		if self.sum >= 0.5:
			return 1
		else:
			return 0

	def df_activation_func(self):
		return 1
