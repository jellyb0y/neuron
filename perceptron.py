from neyron import Neyron, Exceptions

class Perceptron():
	'''
		Класс перцептрона
	'''


	def __init__(self, detectors_dim, neyrons_dim, neyrons_a, layers_count, reacts_dim, reacts_a):
		if not isinstance(detectors_dim, int) \
		or not isinstance(neyrons_dim, int) \
		or not isinstance(layers_count, int) \
		or not isinstance(reacts_dim, int) \
		or not isinstance(neyrons_a, float) \
		or not isinstance(reacts_a, float):
			raise ValueError

		self.detectors_dim = detectors_dim

		if detectors_dim < 1 \
		or neyrons_dim < 1 \
		or layers_count < 1 \
		or reacts_dim < 1:
			raise Exceptions.InvalidInputsCount

		self.reacts = self.gen_neyron_layout(reacts_dim, neyrons_dim, reacts_a)
		self.layers = self.gen_layers(neyrons_dim, detectors_dim, neyrons_a, layers_count)


	def gen_neyron_layout(self, dim, inputs, a):
		layout = []
		for _ in range(dim):
			layout.append(Neyron(inputs, a))

		return layout


	def gen_layers(self, dim, inputs, a, count_layers):
		layers = []
		for i in range(count_layers):
			if i == 0:
				layout = self.gen_neyron_layout(dim, inputs, a)
			else:
				layout = self.gen_neyron_layout(dim, dim, a)
			layers.append(layout)

		return layers

	def calc_x(self, layout, x):
		result = []
		for neyron in layout:
			result.append(neyron.calc(x))

		return result


	def calc(self, x):
		if not isinstance(x, list):
			raise ValueError

		if len(x) != self.detectors_dim:
			raise Exceptions.InvalidDim

		layers = self.layers
		reacts = self.reacts

		for layer in self.layers:
			x = self.calc_x(layer, x)

		result = self.calc_x(reacts, x)

		return result

