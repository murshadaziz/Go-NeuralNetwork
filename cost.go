package main

import (
	"math"
)

func (neuralnetwork NeuralNetwork) Cost(input []float64, label int) float64 {
	output := neuralnetwork.ForwardProgation(input)
	cost := -math.Log(output[label] + 1e-15)
	return cost
}
