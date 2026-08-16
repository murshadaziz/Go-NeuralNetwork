package main

// Loops over all the images one by one and calls the backpropogation method inside to adjust weights
func (neuralnetwork NeuralNetwork) Train(images [][]float64, labels []int) {
	learning_rate := 0.01
	for i, image := range images {
		label := labels[i]
		neuralnetwork.Backpropagation(image, label, learning_rate)
	}
}
