package initialise

import (
	"fmt"
	"math/rand"

	"github.com/sauryagur/neural-network-from-scratch/models/layer"
	"github.com/sauryagur/neural-network-from-scratch/models/neural_network"
	"github.com/sauryagur/neural-network-from-scratch/models/neuron"
)

func CreateMLP() *neural_network.NeuralNetwork {
	fmt.Println("Initialising MLP")

	var nInputs, nOutputs, nHidden int
	nInputs = readPositiveInt("Number of inputs: ")
	nOutputs = readPositiveInt("Number of outputs: ")
	nHidden = readNonNegativeInt("Number of hidden layers: ")
	learningRate := 0.05

	hiddenSizes := make([]int, nHidden)
	for i := 0; i < nHidden; i++ {
		prompt := fmt.Sprintf("Neurons in hidden layer %d: ", i+1)
		hiddenSizes[i] = readPositiveInt(prompt)
	}

	return InitMLP(nInputs, nOutputs, hiddenSizes, learningRate)
}

func InitMLP(nInputs, nOutputs int, hiddenSizes []int, learningRate float64) *neural_network.NeuralNetwork {
	nn := &neural_network.NeuralNetwork{
		Layers: make([]*layer.Layer, 0, 1+len(hiddenSizes)+1),
	}

	nn.Layers = append(nn.Layers, newRandomLayer(nInputs, 0))

	prevSize := nInputs
	for _, size := range hiddenSizes {
		nn.Layers = append(nn.Layers, newRandomLayer(size, prevSize))
		prevSize = size
	}

	nn.Layers = append(nn.Layers, newRandomLayer(nOutputs, prevSize))

	nn.LearningRate = learningRate
	nn.EnableSoftMax = false

	return nn
}

func readPositiveInt(prompt string) int {
	for {
		fmt.Print(prompt)
		var v int
		if _, err := fmt.Scan(&v); err != nil || v <= 0 {
			fmt.Println("Please enter a positive integer.")
			continue
		}
		return v
	}
}

func readNonNegativeInt(prompt string) int {
	for {
		fmt.Print(prompt)
		var v int
		if _, err := fmt.Scan(&v); err != nil || v < 0 {
			fmt.Println("Please enter a non-negative integer.")
			continue
		}
		return v
	}
}

func newRandomNeuron(numWeights int) *neuron.Neuron {
	n := &neuron.Neuron{
		Bias: rand.Float64(),
	}
	n.Weights = make([]float64, numWeights)
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()
	}
	return n
}

func newRandomLayer(numNeurons, weightsPerNeuron int) *layer.Layer {
	l := &layer.Layer{
		Neurons: make([]*neuron.Neuron, numNeurons),
	}
	for i := 0; i < numNeurons; i++ {
		l.Neurons[i] = newRandomNeuron(weightsPerNeuron)
	}
	return l
}
