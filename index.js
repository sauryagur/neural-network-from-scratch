class Neuron {
    constructor(weights, bias) {
        this.weights = weights;
        this.bias = bias;
    }
}

class Layer {
    constructor(neuronsArray) {
        this.neurons = neuronsArray;
    }
}

class NeuralNetwork {
    constructor(layerArray) {
        this.layers = layerArray;
    }
}