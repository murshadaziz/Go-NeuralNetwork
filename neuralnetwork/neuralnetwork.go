package main

import "fmt"

func main() {

	var layer1 Layer
	layer1.init(784, 128, relu)
	var layer2 Layer
	layer2.init(128, 64, relu)
	var layer3 Layer
	layer3.init(64, 10, softmax)

	var neuralnetwork NeuralNetwork
	layers := []Layer{layer1, layer2, layer3}
	neuralnetwork.init(layers, 0.01)

	images, err := readIdxImagesFloat64("dataset/t10k-images.idx3-ubyte")
	if err != nil {
		fmt.Println("error reading images:", err)
		return
	}

	labels, err := readIdxLabelsFloat64("dataset/t10k-labels.idx1-ubyte")
	if err != nil {
		fmt.Println("error reading labels:", err)
		return
	}

	err = neuralnetwork.loadData("data/weights_biases.bin")
	if err != nil {
		fmt.Println("error loading model:", err)
		return
	}
	neuralnetwork.Test(images, labels)

	/*neuralnetwork.Train(images, labels, 5)
	err = neuralnetwork.saveData("data/weights_biases.bin")
	if err != nil {
		fmt.Println("error saving model:", err)
		return
	}*/

}
