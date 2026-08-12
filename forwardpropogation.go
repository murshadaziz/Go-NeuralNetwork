package goneuralnetwork

func (layer *Layer) Inference(input []float64) []float64 {
	output := make([]float64, len(layer.bias))
	for i := range output {
		output[i] = dot(layer.weights[i], input) + layer.bias[i]
	}
	return output
}

func (neuralnetwork *NeuralNetwork) ForwardProgation(input []float64) []float64 {
	current := input
	for _, layer := range *neuralnetwork {
		current = layer.Inference(current)
	}
	return current
}
