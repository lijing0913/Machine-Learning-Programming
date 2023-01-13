## Reservoir Sampling
# select k items from n items with each item having the same probability of being selected, i.e., k/n.

import random as rd
def generator(max): 
	number = 1
	while number < max:
		number += 1
		yield number

# Create as stream generator
stream = generator(10000)

# Doing Reservior Samping from the stream
k = 5
reservoir = []
for i, element in enumerate(stream):
	if i + 1 <= k:
		reservoir.append(element)
	else:
		probability = k / (i + 1)
		if rd.random() < probability:
			# select item in stream and remove one of the k items already selected
			reservoir[rd.choice(range(k))] = element
print(reservoir)