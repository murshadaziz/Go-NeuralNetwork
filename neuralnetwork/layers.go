package neuralnetwork

import (
	"math"
	"math/rand"
)

type NeuralNetwork struct {
	layers       []Layer
	learningRate float64
}

func (neuralnetwork *NeuralNetwork) init(currlayers []Layer, rate float64) {
	neuralnetwork.layers = currlayers
	neuralnetwork.learningRate = rate
}

// Layer struct
type Layer struct {
	weights    [][]float64
	bias       []float64
	activation Activation
	// Caches used for backpropogation
	lastInput []float64 // input this layer received
	delta     []float64 // used for backprob
	output    []float64
}

// Initialises a layer with defined 2d matrix for weights, 1d array for biases and an activation function
func (layer *Layer) Init(inputs, outputs int, act Activation) {
	// Buffer is used so the 2d matrix is contigious
	buffer := make([]float64, inputs*outputs)

	layer.weights = make([][]float64, outputs)

	layer.bias = make([]float64, outputs)

	for i := range layer.weights {
		layer.weights[i] = buffer[i*inputs : (i+1)*inputs]
	}
	layer.lastInput = make([]float64, inputs)

	layer.output = make([]float64, outputs)

	layer.delta = make([]float64, outputs)

	layer.activation = act
}

// Randomizes the weights and biases of all layers of the neural network
func (neuralnetwork NeuralNetwork) Randomise() {
	for i := range neuralnetwork.layers {
		layer := &neuralnetwork.layers[i]
		inputs := len(layer.weights[0])
		scale := math.Sqrt(2.0 / float64(inputs)) // He init, good for ReLU
		for r := range layer.weights {
			for c := range layer.weights[r] {
				layer.weights[r][c] = (rand.Float64()*2 - 1) * scale
			}
		}
	}
}
