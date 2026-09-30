import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	x=np.array(features)
	w= np.array(weights)
	b= np.array(bias)
	y=np.array(labels)

	# expected outputs in raw numbers
	z= np.dot(x,w)+b
	# expected outputs in probabilities
	probabilities= 1/(1+(np.exp(-z)))
	mse= np.mean((probabilities-y)**2)
	return probabilities, mse