from neuron import NeuronDigit
from time import sleep

neuron = NeuronDigit(2, 0.5)

a = [
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1],
]
b = [0, 0, 0, 1]

# def get_bin_arr_4(x, y):
# 	cur_x = bin(x)
# 	cur_y = bin(y)
# 	temp = []
# 	for k in range(4 - (len(cur_x) - 2)):
# 		temp.append(0)
# 	for k in range(len(cur_x) - 2):
# 		temp.append(int(cur_x[k + 2]))

# 	for k in range(4 - (len(cur_y) - 2)):
# 		temp.append(0)
# 	for k in range(len(cur_y) - 2):
# 		temp.append(int(cur_y[k + 2]))

# 	return temp


# for i in range(16):
# 		a.append(get_bin_arr_4(i, i))
# 		if i <= 7:
# 			b.append(0)
# 		else:
# 			b.append(1)

# for i in range(16):
# 		a.append(get_bin_arr_4(i, 15 - i))
# 		if i <= 7:
# 			b.append(0)
# 		else:
# 			b.append(1)

w = neuron.learning(a, b, 0.001)

print('Result: ', w)

for i in range(len(a)):
	print(a[i], neuron.calc(a[i], w), ' - ', b[i])

while True:
	[x, y] = input('[x, y]: ').split(' ')
	x = int(x)
	y = int(y)
	print(neuron.calc([x, y]))
