package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

// Loops over all the images one by one and calls the backpropogation method inside to adjust weights
func (neuralnetwork NeuralNetwork) Train(images [][]float64, labels []int, epochs int) {
	lastLayer := neuralnetwork.layers[len(neuralnetwork.layers)-1]
	output := make([]float64, len(lastLayer.bias)) // sized once, reused every pass
	var label int
	for i := range epochs {
		cost := 0.0
		correct := 0
		fmt.Printf("Epoch %v\n", i)
		for j, image := range images {
			label = labels[j]
			output = neuralnetwork.ForwardProgation(image) // fills output in place
			cost += neuralnetwork.Cost(output, label)
			if argmax(output) == label {
				correct++
			}
			neuralnetwork.Backpropagation(output, label)
		}
		cost /= float64(len(images))
		accuracy := float64(correct) / float64(len(images)) // NEW
		fmt.Printf("Current cost: %v\n", cost)
		fmt.Printf("Accuracy: %.2f%%\n", accuracy*100) // NEW
	}
}

// Saves the current weights and biases in a .bin file
func (neuralnetwork NeuralNetwork) saveData(path string) error {
	// Creates file if it doesnt exists, or truncates it zero bytes and opens it for writing
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// makes a wrapper buffer for *os.file to reduce the syscalls
	// (bufio.Newwriter wraps anything that implements the io.Writer interface that has a Write method)
	// (and returns a *bufio.Writer that also implements the io.Writer interface)
	writer := bufio.NewWriter(f)
	// writes everything from the writer buffer to the .bin file in one go
	defer writer.Flush()
	// iterates over layers
	for _, layer := range neuralnetwork.layers {
		// iterates over rows of layer's weight's
		for i := range layer.weights {
			// iterates over columns
			for j := range layer.weights[i] {
				// writes the weight[i][j] in the writer buffer
				err := binary.Write(writer, binary.BigEndian, layer.weights[i][j])
				if err != nil {
					return err
				}
			}
			// writes the bias[i] in the writer buffer
			err := binary.Write(writer, binary.BigEndian, layer.bias[i])
			if err != nil {
				return err
			}
		}
	}
	// returns err
	return nil
}

// Loads the weights and biases from the .bin file
func (neuralnetwork NeuralNetwork) loadData(path string) error {
	// Opens file ans returns *os.file and error term
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// makes a wrapper buffer for *os.file to reduce the syscalls
	// (bufio.NewReader wraps anything that implements the io.Reader interface that has a Read method)
	// (and returns a *bufio.Reader that also implements the io.Reader interface)
	reader := bufio.NewReader(f)
	for i := range neuralnetwork.layers {
		layer := &neuralnetwork.layers[i]
		for i := range layer.weights {
			for j := range layer.weights[i] {
				// takes the reader and reads 8 byte (float64) into the address of weights[i][j]
				err := binary.Read(reader, binary.BigEndian, &layer.weights[i][j])
				if err != nil {
					return err
				}
			}
			// takes the reader and reads 1 byte into the address of bias[i]
			err := binary.Read(reader, binary.BigEndian, &layer.bias[i])
			if err != nil {
				return err
			}
		}
	}
	return nil
}
