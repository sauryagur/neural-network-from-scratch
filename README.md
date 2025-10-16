# Multilayer Perceptron in Go
#### I built a multilayer perceptron in golang from scratch just to show how easy it is.

## But what the heck is a multilayer perceptron, anyway?

A multilayer perceptron, or MLP, is one of the simplest types of neural networks. It consists of multiple layers, each containing one or more neurons, arranged sequentially from the input layer to the output layer. Every neuron in one layer talks to every neuron in the layer before it and the one after it.

![excalidraw_1.png](.github/images/excalidraw_1.png)

## What's this MLP good for?

MLPs are used for figuring out tricky connections between inputs and outputs. The hidden layers let it spot patterns or features in the input and relate those to your output.

Theoretically, it is possible to model **ANY FUNCTION** using MLPs because of something called the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

MLPs have a lot of applications:
- Regression (guessing numbers)
- Classification (picking categories)
- Even unsupervised learning (just guessing any patterns in general)