package main

import (
	"fmt"

	"github.com/sauryagur/neural-network-from-scratch/mnist"
	"github.com/sauryagur/neural-network-from-scratch/utils/initialise"
)

func main() {
	trainInputs, trainLabels, testInputs, _, err := mnist.LoadMNIST(
		"mnist/data/mnist/train-images.idx3-ubyte",
		"mnist/data/mnist/train-labels.idx1-ubyte",
		"mnist/data/mnist/t10k-images.idx3-ubyte",
		"mnist/data/mnist/t10k-labels.idx1-ubyte",
	)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %d train, %d test\n", len(trainInputs), len(testInputs))

	mlp := initialise.InitMLP(784, 10, []int{128, 64}, 0.05)
	mlp.EnableSoftMax = true

	mlp.Train(trainInputs, trainLabels, 100, 0.01)
}
