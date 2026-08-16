package main

// Gives output of a single layer
func (layer *Layer) Inference(input []float64) []float64 {
	layer.lastInput = input
	output := make([]float64, len(layer.bias))
	for i := range output {
		output[i] = dot(layer.weights[i], input) + layer.bias[i]
	}
	layer.activation(output)

	return output
}

// Gives prediction of the full neural network
func (neuralnetwork NeuralNetwork) ForwardProgation(input []float64) []float64 {
	current := input
	for _, layer := range neuralnetwork {
		current = layer.Inference(current)
	}
	return current
}
