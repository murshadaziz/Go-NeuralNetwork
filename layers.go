package goneuralnetwork

import (
	"math/rand"
)

type NeuralNetwork []Layer

// Layer struct
type Layer struct {
	weights    [][]float64
	bias       []float64
	activation Activation
}

// Initialises a layer with defined 2d matrix for weights, 1d array for biases and an activation function
func (layer *Layer) init(inputs, outputs int, act Activation) {
	// Buffer is used so the 2d matrix is contigious
	buffer := make([]float64, inputs*outputs)

	layer.weights = make([][]float64, outputs)

	layer.bias = make([]float64, outputs)

	for i := range layer.weights {
		layer.weights[i] = buffer[i*inputs : (i+1)*inputs]
	}

	layer.activation = act

}

func (neuralnetwork NeuralNetwork) Randomise() {
	for _, layer := range neuralnetwork {
		for i := range len(layer.weights) {
			for j := range len(layer.weights[0]) {
				layer.weights[i][j] = rand.Float64()*2 - 1
			}
		}
		for i := range len(layer.bias) {
			layer.bias[i] = rand.Float64()*2 - 1
		}
	}
}
