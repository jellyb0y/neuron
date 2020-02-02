from neuron_ import *

class Perceptron():
	"""docstring for Perceptron"""

	def __init__(self, c_inputs, c_layers, d_layers, c_outputs, type_='default'):
		self.c_inputs = c_inputs
		self.c_layers = c_layers
		self.d_layers = d_layers
		self.c_outputs = c_outputs

		self.base_model = NeuronModel
		if type_ == 'digit':
			self.base_model = NeuronDigitModel

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
			layer.append(self.base_model(self.c_inputs))
		self.layers.append(layer)

		for _ in range(1, self.c_layers):
			layer = []
			for _ in range(self.d_layers):
				layer.append(self.base_model(self.d_layers))
			self.layers.append(layer)

		for _ in range(self.c_outputs):
			self.outputs.append(self.base_model(self.d_layers))

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

	def back_propogation(self, input_arr, desired_arr):
		output_arr = self.calc(input_arr);

		for index, output in enumerate(output_arr):
			dif = desired_arr[index] - output

			curent_output = self.outputs[index]

			sum_weight = None
			for weight in curent_output.inputs_weights:
				sum_weight += weight

			
