package neural_network

import (
	"errors"
	"fmt"

	"github.com/sauryagur/neural-network-from-scratch/models/layer"
	"github.com/sauryagur/neural-network-from-scratch/utils/activations"
)

type NeuralNetwork struct {
	Layers       []*layer.Layer
	LearningRate float64
}

func (net *NeuralNetwork) Output(prev []float64) ([]float64, error) {
	for i := range net.Layers {
		next, err := net.Layers[i].Forward(prev)
		net.Layers[i].LastOutput = next
		if err != nil {
			return nil, errors.New("bad input")
		}
		prev = next
	}
	return prev, nil
}

func (net *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i := 0; i < len(inputs); i++ {
			x := inputs[i]
			// yTrue = the list of actual expected outputs
			yTrue := targets[i]

			// yPred = the list of predicted outputs that we get from forward propagation
			// we input x in the network which is the collection/array of inputs for which we expect the output yTrue
			yPred, _ := net.Output(x)

			// we need to calculate how wrong the model is right now
			loss := CalculateLoss(yPred, yTrue)

			// this totalLoss thing is just for diagnostics
			totalLoss += loss

			// calculate gradients by backpropagation and store them in the layers
			net.Backward(yTrue)

			// now update each layer according to the learningRate and stored gradients
			net.UpdateParameters(learningRate)
		}

		fmt.Printf("Epoch %d: Average Loss = %f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

// Backpropagate and store the gradients of each layer in the layer itself
func (net *NeuralNetwork) Backward(yTrue []float64) {
	var deltaNext []float64

	// counting backwards layer by layer
	for i := len(net.Layers) - 1; i >= 0; i-- {
		currentLayer := net.Layers[i]

		// if current layer is output layer then calculate the deltaNext
		if i == len(net.Layers)-1 {
			deltaNext = make([]float64, len(yTrue))
			// calculate delta for each neuron
			for j := 0; j < len(yTrue); j++ {
				// here a is last output of the output layer
				a := currentLayer.LastOutput[j]
				deltaNext[j] = (a - yTrue[j]) * activations.SigmoidDerivative(a)
			}
		} else {
			// Hidden layer: error = (W_next^T * deltaNext) * activation derivative
			nextLayer := net.Layers[i+1]
			delta := make([]float64, len(currentLayer.LastOutput))
			for j := range delta {
				sum := 0.0
				for k := range nextLayer.Neurons {
					sum += nextLayer.Neurons[k].Weights[j] * deltaNext[k]
				}
				delta[j] = sum * activations.SigmoidDerivative(currentLayer.LastOutput[j])
			}
			deltaNext = delta
		}

		// calculate gradients for current layer weights and biases
		for j, neuron := range currentLayer.Neurons {
			// for each neuron, find the delta
			delta := deltaNext[j]
			// for each weight in the neuron
			for k := range neuron.Weights {
				inputActivation := 0.0
				if i == 0 {
					// input layer inputs
					inputActivation = currentLayer.Inputs[k]
				} else {
					inputActivation = net.Layers[i-1].LastOutput[k]
				}

				// weight gradient of current layer = delta * input activation of current layer
				neuron.WeightGradients[k] = delta * inputActivation
			}
			// gradient for bias = delta
			neuron.BiasGradient = delta
		}
	}
}

// update Parameters
func (net *NeuralNetwork) UpdateParameters(learningRate float64) {
	for _, currentLayer := range net.Layers {
		for _, neuron := range currentLayer.Neurons {
			for i := range neuron.Weights {
				neuron.Weights[i] -= learningRate * neuron.WeightGradients[i]
			}
			neuron.Bias -= learningRate * neuron.BiasGradient
		}
	}
}
func CalculateLoss(yPred []float64, yTrue []float64) float64 {
	if len(yPred) != len(yTrue) {
		panic("Lengths of predictions and true values must match")
	}
	var sum float64 = 0
	for i := 0; i < len(yPred); i++ {
		diff := yTrue[i] - yPred[i]
		sum += 0.5 * diff * diff
	}
	return sum / float64(len(yPred)) // average loss per output neuron
}
