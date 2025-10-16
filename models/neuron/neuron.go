package neuron

import (
	"errors"

	"github.com/sauryagur/neural-network-from-scratch/utils/activations"
)

type Neuron struct {
	Weights         []float64
	WeightGradients []float64
	Bias            float64
}

func (neuron Neuron) Forward(inputs []float64) (float64, error) {
	if len(inputs) != len(neuron.Weights) {
		return 0.0, errors.New("invalid input size")
	}

	var sum float64
	for i, w := range neuron.Weights {
		sum += w * inputs[i]
	}
	sum += neuron.Bias

	return activations.Sigmoid(sum), nil
}
