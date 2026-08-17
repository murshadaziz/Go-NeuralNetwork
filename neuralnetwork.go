package main

import "fmt"

func main() {

	var layer1 Layer
	layer1.init(4, 3, relu, false)
	var layer2 Layer
	layer2.init(3, 2, softmax, false)

	var neuralnetwork NeuralNetwork
	neuralnetwork = append(neuralnetwork, layer1)
	neuralnetwork = append(neuralnetwork, layer2)

	err := neuralnetwork.loadData("data/weights_biases.bin")
	input := []float64{1, 2, 3, 4}
	output := neuralnetwork.ForwardProgation(input)
	fmt.Println(output)
	if err != nil {
		return
	}

	/*
		images, err := readIdxImagesFloat64("dataset/train-images.idx3-ubyte")
		if err != nil {
			fmt.Println("error reading images:", err)
			return
		}

		labels, err := readIdxLabelsFloat64("dataset/train-labels.idx1-ubyte")
		if err != nil {
			fmt.Println("error reading labels:", err)
			return
		}

		fmt.Println("num images:", len(images))
		fmt.Println("num labels:", len(labels))
		fmt.Println("pixels per image:", len(images[0]))

		fmt.Println("first label:", labels[0])
		fmt.Println("first image (first 28 pixels):")
		for i := 0; i < 784; i++ {
			fmt.Printf("%.1f ", images[0][i])
		}
		fmt.Println()
	*/
}
