from perceptron_ import Perceptron

samples = {
    (0, 1, 1, 1): (0, 1, 1, 1),
    (1, 0, 1, 1): (1, 0, 1, 1),
    (1, 1, 0, 1): (1, 1, 0, 1),
    (1, 1, 1, 0): (1, 1, 1, 0)
}

perceptron = Perceptron(4, 1, 4, 4, 'digit')

perceptron.learn(samples, trace_flag=True, learning_constant=1.0)

for input_arr, desired_arr in samples.items():
    result = perceptron.calc(input_arr)
    print(result, desired_arr)
