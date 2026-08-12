package goneuralnetwork

type NeuralNetwork []Layer

// Layer struct
type Layer struct {
	inputs     []float64
	output     []float64
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
