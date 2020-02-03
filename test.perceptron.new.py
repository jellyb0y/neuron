from perceptron_ import Perceptron

input_arr = [0, 1, 0, 1, 0]
desired_arr = [1, 0, 1, 0, 1]

perceptron = Perceptron(5, 1, 1, 5, 'digit')
result = perceptron.calc(input_arr)

print(result)

perceptron.learn(input_arr, desired_arr, trace_flag=True, delta=0.001)

result = perceptron.calc(input_arr)

print(result)