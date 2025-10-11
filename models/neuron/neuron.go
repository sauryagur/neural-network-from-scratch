package neuron

import (
	"errors"
	"math"
)

type Neuron struct {
	Weights	[]float64
	Bias	float64
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
	
	return sigmoid(sum), nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0 * x))
}
