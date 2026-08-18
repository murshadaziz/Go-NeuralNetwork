package main

import (
	"math"
)

type Activation func([]float64) []float64

func relu(input []float64) []float64 {
	for i := 0; i < len(input); i++ {
		input[i] = math.Max(0, input[i])
	}
	return input
}

func linear(input []float64) []float64 {
	return input
}

func softmax(input []float64) []float64 {
	max := input[0]
	for i := range input {
		if input[i] > max {
			max = input[i]
		}
	}

	sum := 0.0
	for i := range input {
		input[i] = math.Exp(input[i] - max) // overwrite in place with exp value
		sum += input[i]
	}

	for i := range input {
		input[i] /= sum // normalize in place
	}

	return input
}
