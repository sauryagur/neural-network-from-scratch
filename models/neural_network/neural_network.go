package neural_network

import (
	"errors"
	"github.com/sauryagur/neural-network-from-scratch/models/layer"
	"math"
	)
	
type NeuralNetwork struct {
	Layers 		[]*layer.Layer
	LearningRate	float
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
			yTrue := targets[i]

			yPred, _ := net.Output(x)

			loss := CalculateLoss(yPred, yTrue)
			totalLoss += loss
			
			net.Backward(yTrue)

			net.UpdateParameters(learningRate)
		}

		fmt.Printf("Epoch %d: Average Loss = %f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

func (net *NeuralNetwork) CalculateLoss(yPred []float64, yTrue []float64) []float64 {
	output := make([]float64, 0, len(yPred))
	for i := 0; i < len(yPred); i++ {
		output = append(output, 0.5 * math.Pow(yTrue[i] - yPred[i], 2))
	}
	return output
}
