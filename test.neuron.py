from neyron import Neyron
from time import sleep

neyron = Neyron(3, 1.0)

a = [
	[1, 1, 0],
	[1, 0, 1],
	[1, 1, 1]
]

b = [0, 0, 1]

def sum_miss(w=None):
	i = 0
	sum_ = 0
	for k in a:
		result = neyron.calc(k, w)
		miss = (b[i] - result)**2
		i += 1
		sum_ += miss
 
	return sum_

w = neyron.w
acc = 0.000001
last = 0
cur = 100
while cur**2 > 0.001:
	cur = sum_miss(w)
	print(cur, w)
	for i in range(len(w)):
		direct = b[i] - neyron.calc(a[i], w)
		w[i] += acc * direct

	last = cur

print('Result: ', w)
