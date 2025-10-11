package layer

import (
	"errors"
	"models/neuron"
	)

type Layer struct {
	Neurons	[]*neuron.Neuron
}

func (layer *Layer) Forward(inputs []float64) ([]float64, error) {
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
	
	return output, nil
}
