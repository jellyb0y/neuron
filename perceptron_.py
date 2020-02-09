from neuron_ import *

class Perceptron():
	"""docstring for Perceptron"""

	def __init__(\
		self,\
		c_inputs,\
		c_layers,\
		d_layers,\
		c_outputs,\
		type_='default'\
	):
		self.c_inputs = c_inputs
		self.c_layers = c_layers
		self.d_layers = d_layers
		self.c_outputs = c_outputs

		self.base_output_model = NeuronModel
		if type_ == 'digit':
			self.base_output_model = NeuronDigitModel

		self.inputs = []
		self.layers = []
		self.outputs = []
		
		self.generate_neurons()
		self.tie_layers()

	def generate_neurons(self):
		for _ in range(self.c_inputs):
			self.inputs.append(EmptyNeuron())

		layer = []
		for _ in range(self.d_layers):
			layer.append(NeuronModel(self.c_inputs))
		self.layers.append(layer)

		for _ in range(1, self.c_layers):
			layer = []
			for _ in range(self.d_layers):
				layer.append(NeuronModel(self.d_layers))
			self.layers.append(layer)

		for _ in range(self.c_outputs):
			self.outputs.append(self.base_output_model(self.d_layers))

	def tie_layers(self):
		first_layer = self.layers[0]

		for input_ in self.inputs:
			for neuron in first_layer:
				input_.set_output(neuron)
				neuron.set_input(input_)

		for index, layer in enumerate(self.layers[1:]):
			for neuron in layer:
				for last_neuron in self.layers[index]:
					neuron.set_input(last_neuron)
					last_neuron.set_output(neuron)

		last_layer = self.layers[len(self.layers) - 1]
		for output in self.outputs:
			for last_neuron in last_layer:
				output.set_input(last_neuron)
				last_neuron.set_output(output)

	def calc(self, input_arr):
		for index, input_ in enumerate(self.inputs):
			input_.set_value(input_arr[index])

		output_arr = []
		for output in self.outputs:
			output_arr.append(output.value)

		return output_arr

	def back_propagation(self, input_arr, desired_arr):
		output_arr = self.calc(input_arr)

		sum_error = 0

		for index, output in enumerate(output_arr):
			dif = output - desired_arr[index]
			sum_error += abs(dif)
			curent_output = self.outputs[index]
			self.propagation(curent_output, dif)

		return sum_error

	def propagation(self, neuron, dif):
		if isinstance(neuron, EmptyNeuron):
			return None

		df = neuron.df_activation_func()
		weight_delta = dif * df

		for index, _ in neuron.inputs_weights.items():
			inputs_neuron = neuron.inputs[index]
			value = inputs_neuron.value
			neuron.inputs_weights[index] -= weight_delta * self.learning_constant * value

			func_error = neuron.inputs_weights[index] * dif
			self.propagation(inputs_neuron, func_error)

	def learn(\
		self,\
		samples,\
		learning_constant=0.1,\
		trace_flag=False,\
		stop_condition='delta',\
		delta=0.0001\
	):
		if not isinstance(delta, float):
			raise Exceptions.InvalidValue

		self.learning_constant = learning_constant
		self.delta = delta

		error = None
		if stop_condition == 'delta':
			while (error == None or error > delta):
				error = 0

				for input_arr, desired_arr in samples.items():
					error += abs(self.back_propagation(input_arr, desired_arr))

				if trace_flag:
					print(error)
		elif isinstance(stop_condition, int):
			for _ in range(stop_condition):
				error = 0

				for input_arr, desired_arr in samples.items():
					error += abs(self.back_propagation(input_arr, desired_arr))

				if trace_flag:
					print(error)
		else:
			raise Exceptions.InvalidValue