package neural_network

import (
	"errors"
	"github.com/sauryagur/neural-network-from-scratch/models/layer"
	)
	
type NeuralNetwork struct {
	Layers []*layer.Layer
}

func (net *NeuralNetwork) Output(prev []float64) ([]float64, error) {
	for i, _ := range net.Layers {
		next, err := net.Layers[i].Forward(prev)
		if err != nil {
			return nil, errors.New("bad input")
		}
		prev = next
	}
	return prev, nil
}
