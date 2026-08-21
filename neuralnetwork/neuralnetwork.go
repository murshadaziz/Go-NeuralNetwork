package neuralnetwork

import (
	"fmt"
	"os"
	"path/filepath"
)

func ProjectRoot() (string, error) {
	// Gets working directory of the file that calls it
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	// inifite loop
	for {
		// os.Stat finds info of the file specified and returns err != nil then it means file doesnt exist
		// filepath.Join joins the go.mod to current directory
		_, err := os.Stat(filepath.Join(dir, "go.mod"))
		if err == nil {
			return dir, nil
		}
		// filepath.Dir returns the parent directory of current directory
		parent := filepath.Dir(dir)
		// if traverses to root but still doesnt find go.mod
		if parent == dir {
			return "", fmt.Errorf("go.mod not found from %s upward", dir)
		}
		// checks again in the parent for go.mod
		dir = parent
	}
}

func Run() {
	// gets the root directory of project
	root, err := ProjectRoot()
	if err != nil {
		fmt.Println("error resolving project root:", err)
		return
	}

	var layer1 Layer
	layer1.Init(784, 128, Relu)
	var layer2 Layer
	layer2.Init(128, 64, Relu)
	var layer3 Layer
	layer3.Init(64, 10, Softmax)

	var neuralnetwork NeuralNetwork
	layers := []Layer{layer1, layer2, layer3}
	neuralnetwork.init(layers, 0.01)
	imagesPath := filepath.Join(root, "dataset", "t10k-images.idx3-ubyte")
	labelsPath := filepath.Join(root, "dataset", "t10k-labels.idx1-ubyte")
	weightsPath := filepath.Join(root, "data", "weights_biases.bin")
	images, err := ReadIdxImagesFloat64(imagesPath)
	if err != nil {
		fmt.Println("error reading images:", err)
		return
	}

	labels, err := ReadIdxLabelsFloat64(labelsPath)
	if err != nil {
		fmt.Println("error reading labels:", err)
		return
	}

	err = neuralnetwork.LoadData(weightsPath)
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
