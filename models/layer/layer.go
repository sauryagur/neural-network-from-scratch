package layer

import (
	"errors"

	"github.com/sauryagur/neural-network-from-scratch/models/neuron"
)

type Layer struct {
	Neurons    []*neuron.Neuron
	LastOutput []float64
	Inputs     []float64 // this is used in backprop when this layer is the input layer
}

func (layer *Layer) Forward(inputs []float64) ([]float64, error) {
	layer.Inputs = inputs
	output := make([]float64, 0, len(layer.Neurons))

	for i := range layer.Neurons {
		// for each neuron
		neuronOutput, err := layer.Neurons[i].Forward(inputs)
		// pass in all outputs
		if err != nil {
			// if any neuron throws an error
			return nil, errors.New("input size incompatible with layer")
		}
		output = append(output, neuronOutput)
	}
	layer.LastOutput = output
	return output, nil
}
