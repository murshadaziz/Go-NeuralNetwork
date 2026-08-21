package main

// Gives output of a single layer
func (layer *Layer) Inference(input []float64) {
	layer.lastInput = input
	for i := range layer.output {
		layer.output[i] = dot(layer.weights[i], input) + layer.bias[i]
	}
	layer.activation(layer.output)

}

// Gives prediction of the full neural network
func (neuralnetwork NeuralNetwork) ForwardProgation(input []float64) {
	current := input
	for i := range neuralnetwork.layers {
		layer := &neuralnetwork.layers[i]
		layer.Inference(current)
		current = layer.output
	}
}

// returns the index of the largest value in a slice — i.e. the predicted class
func argmax(values []float64) int {
	best := 0
	for i := range values {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
