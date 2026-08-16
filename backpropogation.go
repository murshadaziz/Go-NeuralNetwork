package main

import (
	"math"
)

// For softmax
func (neuralnetwork NeuralNetwork) Cost(input []float64, label int) float64 {
	output := neuralnetwork.ForwardProgation(input)
	cost := -math.Log(output[label] + 1e-15)
	return cost
}

// Backpropagation has the following steps:
//
//  1. Calculate the delta of the last layer using the cost function which is cross entropy for softmax
//     Delta has the same size as the number of neurons in that layer.
//
//  2. Calculate the weight gradients by multiplying delta with
//     the input/activation from the previous layer.
//     dW = delta * previousActivation^T
//
// 3. Calculate the delta for the previous layer by:
//
//   - summing the weights multiplied by the current delta
//
//   - multiplying by the derivative of the previous layer's activation
//
//     delta_prev = sum(weight * delta) * activationDerivative
//
//     4. For ReLU:
//     if the previous layer's z > 0:
//     delta_prev = sum(weight * delta)
//     else:
//     delta_prev = 0
func (neuralnetwork NeuralNetwork) Backpropagation(input []float64, label int, lr float64) {
	// forward pass, caches lastInput/lastOutput on every layer
	output := neuralnetwork.ForwardProgation(input)
	// Number of outputs
	numClasses := len(output)
	target := make([]float64, numClasses)
	// sets 1.0 for correct label rest remain 0.0
	target[label] = 1.0

	// delta for output layer: softmax + cross-entropy simplifies to (output - target)
	// delta will have size equal the neurons in the current layer
	delta := make([]float64, numClasses)
	for i := range delta {
		// sets delta one by one
		delta[i] = output[i] - target[i]
	}

	// walk backwards through layers
	for l := len(neuralnetwork) - 1; l >= 0; l-- {
		// gets one layer
		layer := &neuralnetwork[l]

		// compute delta for the PREVIOUS layer before we overwrite this layer's weights
		var prevDelta []float64
		// doesnt compute for last layer
		if l > 0 {
			// makes a slice equal to length of neurons in the prev layer
			prevDelta = make([]float64, len(layer.lastInput))
			// iterates over the prev delta slice
			for j := range prevDelta {
				sum := 0.0
				// adds all the weights*delta of the all the neurons of current layer
				// here i is the neuron number and j is the lastinput aka the neurons in the previous layer
				for i := range delta {
					sum += layer.weights[i][j] * delta[i]
				}
				// ReLU derivative: 1 if this layer's input (prev layer's output) was > 0
				// as j was the previous layers neurons so we set the prevdelta[j] as sum
				// if the derivative of last is not zero
				if layer.lastInput[j] > 0 {
					prevDelta[j] = sum
				} else {
					prevDelta[j] = 0
				}
			}
		}

		// update weights and biases using current delta
		// i is number of neurons in current layer
		// j is the number of neurons in previous layer aka number of inputs aka number of weights per neuron
		for i := range layer.weights {
			for j := range layer.weights[i] {
				// updates using learning rate lr
				layer.weights[i][j] -= lr * delta[i] * layer.lastInput[j]
			}
			layer.bias[i] -= lr * delta[i]
		}
		// sets delta for the previous layer
		delta = prevDelta
	}
}
