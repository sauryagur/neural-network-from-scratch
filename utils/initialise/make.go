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
	var numberOfInputs, numberOfOutputs, numberOfHiddenLayers int
	fmt.Printf("Number of inputs: ")
	if _, err := fmt.Scan(&numberOfInputs); err != nil {
		panic(err)
	}
	fmt.Printf("Number of outputs: ")
	if _, err := fmt.Scan(&numberOfOutputs); err != nil {
		panic(err)
	}
	fmt.Printf("Number of hidden layers: ")
	if _, err := fmt.Scan(&numberOfHiddenLayers); err != nil {
		panic(err)
	}

	var hiddenLayerNeurons []int

	for i := 0; i < numberOfHiddenLayers; i++ {
		fmt.Printf("Neurons in layer %d: ", i+1)
		if _, err := fmt.Scan(&hiddenLayerNeurons[i]); err != nil {
			panic(err)
		}
	}
	mlp := initMLP(numberOfInputs, numberOfOutputs, hiddenLayerNeurons)
	return mlp
}

func initMLP(inputNeurons, outputNeurons int, hiddenLayerNeurons []int) *neural_network.NeuralNetwork {
	mlp := new(neural_network.NeuralNetwork)

	// for input layer
	mlp.Layers[0] = randomLayer(inputNeurons, 1)

	// for the rest of the hidden layers
	for hiddenLayer := range hiddenLayerNeurons {
		lastLayer := mlp.Layers[len(mlp.Layers)-1]
		lastLayerNeurons := len(lastLayer.Neurons)
		mlp.Layers = append(mlp.Layers, randomLayer(lastLayerNeurons, hiddenLayerNeurons[hiddenLayer]))
	}

	lastLayerWeights := len(mlp.Layers[len(mlp.Layers)-1].Neurons)
	mlp.Layers[len(mlp.Layers)] = randomLayer(outputNeurons, lastLayerWeights)

	return mlp
}

func randomNeuron(weights int) *neuron.Neuron {
	outputNeuron := new(neuron.Neuron)
	outputNeuron.Bias = rand.Float64()
	for i := 0; i < weights; i++ {
		outputNeuron.Weights = append(outputNeuron.Weights, rand.Float64())
	}
	return outputNeuron
}

func randomLayer(size, weights int) *layer.Layer {
	outputLayer := new(layer.Layer)
	for i := 0; i < size; i++ {
		outputLayer.Neurons = append(outputLayer.Neurons, randomNeuron(weights))
	}
	return outputLayer
}
