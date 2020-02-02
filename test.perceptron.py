from perceptron import PerceptronDigit
from time import sleep

perceptron = PerceptronDigit(2, 1, 0.5, 1, 1, 0.5)

a = [
	[1, 0],
	[0, 1],
	[1, 1],
]

b = [
	[0,],
	[0,],
	[1,]
]

error = perceptron.learning(a, b, 0.01, log=True)
print('Result: ', error)

weight = perceptron.give_weights()
print(weight)
	
result = []
for l in a:
	result.append(perceptron.calc(l))
print(a, result)

# [
# 	[
# 		[0.3692455023544903, 0.10921072494139583, 0.7351132777134908],
# 		[0.7745473415299304, 0.25920369068187055, 0.19595089738124916]
# 	],
# 	[
# 		[0.43693214827662397, 0.6144849542019204, 0.08035843534120901],
# 		[0.3260619338708298, 0.3758459839410947, 0.31265737917307246]
# 	],
# 	[
# 		[1.0995439211519122, 0.04318779012855534, 0.30398899867840035],
# 		[1.0994993282239687, 0.33304856072998834, 0.35162892655672706]
# 	]
# ]