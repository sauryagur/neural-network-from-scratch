function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    let s = sigmoid(x);
    return s * (1 - s); // im lazy lol
}

class Neuron {
    constructor(weights, bias) {
        this.weights = weights;
        this.bias = bias;
    }

    output(neuronInputs) {
        try {
            let sum = this.bias;
            neuronInputs.forEach((neuronInput, index) => {
                sum += neuronInput * this.weights[index]; //z = bias + w1x1 + w2x2 + blah blah blah
            });
            return sigmoid(sum); // im using sigmoid as activation function bc i wanna implement logic gates

        } catch (error) {
            console.error("error in neuron output: " + error);
        }
    }
}

class Layer {
    constructor(neuronsArray) {
        this.neurons = neuronsArray;
    }
    output(neuronInputs) {
        try {
            return this.neurons.map(neuron => neuron.output(neuronInputs)); // put the neuron inputs array through each neuron and put the output in an array and return it
        } catch (error) {
            console.error("error in layer: " + error);
        }
    }
}

class NeuralNetwork {
    constructor(layerArray) {
        this.layers = layerArray;
    }

    predict(inputVector) {
        let previousInput = inputVector; // for first layer previous input is just the input
        try {
            for (let i = 0; i < this.layers.length; i++) {
                let currentLayer = this.layers[i]; //iterate through each layer
                previousInput = currentLayer.output(previousInput); // pass the output of the previous layer to current layer and store the output in previous input
            }
            return previousInput; // return the output from the final layer and call it prediction

        } catch (error) {
            console.error("error in layer: " + error);
        }
    }
}