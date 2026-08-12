package main

import (
	"fmt"
)

func main() {
	var layer1 Layer
	layer1.init(4, 3, relu)
	var layer2 Layer
	layer2.init(3, 2, softmax)

	var neuralnetwork NeuralNetwork
	neuralnetwork = append(neuralnetwork, layer1)
	neuralnetwork = append(neuralnetwork, layer2)

	neuralnetwork.Randomise()

	input := []float64{1, 2, 3, 4}
	output := neuralnetwork.ForwardProgation(input)

	fmt.Println(output)

}
