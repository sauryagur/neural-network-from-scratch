package activations

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func SoftmaxVector(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}

	// find max
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}

	// compute exponentials & sum
	expSum := 0.0
	exps := make([]float64, len(logits))
	for i, v := range logits {
		exps[i] = math.Exp(v - max)
		expSum += exps[i]
	}

	// normalise to probabilities
	probs := make([]float64, len(logits))
	for i := range exps {
		probs[i] = exps[i] / expSum
	}
	return probs
}
