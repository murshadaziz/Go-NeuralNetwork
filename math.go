package goneuralnetwork

func dot(a, b []float64) float64 {
	var sum float64

	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}
