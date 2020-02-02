from perceptron_ import Perceptron

perceptron = Perceptron(3, 2, 2, 2, 'digit')
result = perceptron.calc([1, 2, 0])

print(result)