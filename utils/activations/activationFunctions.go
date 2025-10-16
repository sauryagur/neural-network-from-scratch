package activations

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
