from neyron import NeuronDigit
from time import sleep

neyron = NeuronDigit(4, 1.0)

a = [
	[1, 1, 1, 1],
	[1, 1, 0, 1],
	[0, 1, 1, 0],
	[0, 1, 1, 1],
	[0, 1, 0, 0],
	[0, 1, 0, 1],
	[1, 1, 1, 0],
	[1, 1, 1, 1],
	[0, 0, 1, 0],
	[0, 0, 1, 1],
	[1, 0, 0, 0],
	[1, 0, 0, 1],
	[1, 0, 1, 0],
	[1, 0, 1, 1],
	[0, 0, 0, 0],
	[0, 0, 0, 1],
	[1, -1, 0, 0],
	[1, -1, 0, 1],
	[0, -1, 1, 0],
	[0, -1, 1, 1],
	[0, -1, 0, 0],
	[0, -1, 0, 1],
	[1, -1, 1, 0],
	[1, -1, 1, 1],
]

b = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]

def learn(a, b, w):
	learning_constant = 0.0001
	error = b - neyron.calc(a, w)
	for i in range(len(w)):
		if i == 0:
			w[0] += error * learning_constant
		else: 
			w[i] += error * learning_constant * a[i - 1]

	return w


def sum_miss(w=None):
	sum_ = 0
	for i in range(len(a)):
		result = neyron.calc(a[i], w)
		miss = (b[i] - result)**2
		sum_ += miss
 
	return sum_


def learning(w):
	error = None
	while error != 0:
		for i in range(len(a)):
			w = learn(a[i], b[i], w)
		error = sum_miss(w)
		print(error, w)
	return w

w = learning(neyron.weights)

print('Result: ', w)

for i in range(len(a)):
	print(a[i], neyron.calc(a[i], w), ' - ', b[i])

while True:
	[x, y, z, f] = input('[x, y, z, f]: ').split(' ')
	x = int(x)
	y = int(y)
	z = int(z)
	f = int(f)
	print(neyron.calc([x, y, z, f]))
