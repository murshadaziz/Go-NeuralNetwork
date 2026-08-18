package main

import (
	"math"
)

// For softmax
func (neuralnetwork NeuralNetwork) Cost(output []float64, label int) float64 {
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
func (neuralnetwork NeuralNetwork) Backpropagation(output []float64, label int) {
	// gets last layer of neural network
	lastLayer := &neuralnetwork.layers[len(neuralnetwork.layers)-1]
	copy(lastLayer.delta, output)
	// delta for output layer: softmax + cross-entropy simplifies to (output - target)
	// delta will have size equal the neurons in the current layer
	// as all the values in target are 0 except the label
	lastLayer.delta[label] -= 1.0

	// walk backwards through layers
	for l := len(neuralnetwork.layers) - 1; l >= 0; l-- {
		layer := &neuralnetwork.layers[l]
		// calculates prev delta for all layers except the first
		if l > 0 {
			computePrevDelta(layer, &neuralnetwork.layers[l-1])
		}

		accumulateGradients(layer)
	}
}

// Takes current layer and previous layer as input and calculates the delta for previous layer
func computePrevDelta(layer, prevLayer *Layer) {
	// adds all the weights*delta of the all the neurons of current layer
	// here i is the neuron number and j is the lastinput aka the neurons in the previous layer
	for j := range prevLayer.delta {
		sum := 0.0
		for i := range layer.delta {
			sum += layer.weights[i][j] * layer.delta[i]
		}
		// ReLU derivative: 1 if this layer's input (prev layer's output) was > 0
		// as j was the previous layers neurons so we set the prevdelta[j] as sum
		// if the derivative of last is not zero
		if layer.lastInput[j] > 0 {
			prevLayer.delta[j] = sum
		} else {
			prevLayer.delta[j] = 0
		}
	}
}

// Accumulates the gradients of a layer using its delta and lastinputs
func accumulateGradients(layer *Layer) {
	// i is number of neurons in current layer
	// j is the number of neurons in previous layer aka number of inputs aka number of weights per neuron
	for i := range layer.weights {
		for j := range layer.weights[i] {
			layer.weightGrad[i][j] += layer.delta[i] * layer.lastInput[j]
		}
		layer.biasGrad[i] += layer.delta[i]
	}
}

// Applies the gradients to the weights and biases of the layer after accumulating for batchsize
func applyGradients(layer *Layer, lr float64, batchSize int) {
	for i := range layer.weights {
		for j := range layer.weights[i] {
			layer.weights[i][j] -= lr * layer.weightGrad[i][j] / float64(batchSize)
			layer.weightGrad[i][j] = 0
		}
		layer.bias[i] -= lr * layer.biasGrad[i] / float64(batchSize)
		layer.biasGrad[i] = 0
	}
}
