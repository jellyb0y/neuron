from neuron_ import  *

input_array = [1, 0, 0]
neuron = NeuronDigitModel(3)

for i in range(3):
	input_ = EmptyNeuron()
	input_.set_output(neuron)
	neuron.set_input(input_)
	neuron.set_value(input_, input_array[i])

print(neuron.value)