"use strict";
var Util_1 = require('./Util');
var NeuronLayer_1 = require('./NeuronLayer');
var TrainingData_1 = require('./TrainingData');
function def(thing) {
    return typeof thing !== "undefined";
}
var NeuralNetwork = (function () {
    function NeuralNetwork(args) {
        this.numInputs = 9;
        this.numHiddenLayers = 2;
        this.neuronsPerHiddenLayer = 6;
        this.numOutputs = 5;
        this.networkEvaluationSuccessThreshold = 0.01;
        this.learningRateEta = 0.01;
        if (!args)
            args = {};
        if (def(args.numInputs))
            this.numInputs = args.numInputs;
        if (def(args.numHiddenLayers))
            this.numHiddenLayers = args.numHiddenLayers;
        if (def(args.neuronsPerHiddenLayer))
            this.neuronsPerHiddenLayer = args.neuronsPerHiddenLayer;
        if (def(args.numOutputs))
            this.numOutputs = args.numOutputs;
        if (def(args.networkEvaluationSuccessThreshold))
            this.networkEvaluationSuccessThreshold = args.networkEvaluationSuccessThreshold;
        if (def(args.learningRateEta))
            this.learningRateEta = args.learningRateEta;
        this.layers = [];
        var inputs = this.numInputs;
        for (var i = 0; i < this.numHiddenLayers; ++i) {
            this.layers.push(new NeuronLayer_1.NeuronLayer(inputs, this.neuronsPerHiddenLayer));
            inputs = this.neuronsPerHiddenLayer;
        }
        this.layers.push(new NeuronLayer_1.NeuronLayer(inputs, this.numOutputs));
    }
    NeuralNetwork.prototype.feedForward = function (inputs) {
        var outputs = inputs.slice();
        for (var i = 0; i < this.layers.length; ++i) {
            outputs = this.layers[i].feedForward(outputs);
        }
        return outputs;
    };
    NeuralNetwork.prototype.export = function (weights, offset) {
        for (var i = 0; i < this.numInputs; ++i) {
            offset = this.layers[i].export(weights, offset);
        }
        return offset;
    };
    NeuralNetwork.prototype.import = function (weights, offset) {
        for (var i = 0; i < this.numInputs; ++i) {
            offset = this.layers[i].import(weights, offset);
        }
        return offset;
    };
    NeuralNetwork.prototype.evaluate = function (testData, threshold) {
        threshold = threshold || this.networkEvaluationSuccessThreshold;
        var successes = 0;
        for (var i = 0; i < testData.data.length; ++i) {
            var result = this.evaluatePair(testData.data[i], threshold);
            if (result < 0)
                return result;
            successes += result;
        }
        return successes;
    };
    NeuralNetwork.prototype.evaluatePair = function (testPair, threshold) {
        threshold = threshold || this.networkEvaluationSuccessThreshold;
        var output = this.feedForward(testPair.input);
        var amount = output.length;
        if (amount != testPair.output.length)
            return -1;
        for (var i = 0; i < amount; ++i) {
            var diff = testPair.output[i] - output[i];
            if (Math.abs(diff) > threshold)
                return 0;
        }
        return 1;
    };
    NeuralNetwork.prototype.stochasticGradientDescent = function (trainingData, epochs, miniBatchSize, eta, printer, testData) {
        var _this = this;
        var doEvaluation = typeof (printer) == "function"
            && (testData instanceof TrainingData_1.TrainingData)
            && testData.data.length > 0;
        var i = 0;
        var loop = function () {
            trainingData.reset();
            for (var j = 0; j < trainingData.data.length / miniBatchSize; ++j) {
                _this.updateMiniBatch(trainingData, miniBatchSize, eta);
            }
            if (doEvaluation) {
                _this.logEpochEvaluation(i, printer, testData);
            }
            i++;
            if (i < epochs)
                setTimeout(loop, 0);
        };
        setTimeout(loop, 0);
    };
    NeuralNetwork.prototype.logEpochEvaluation = function (epoch, printer, testData) {
        console.log("Epoch: " + epoch + ": " + this.evaluate(testData) + " / " + testData.data.length);
        printer("Epoch: " + epoch + ": " + this.evaluate(testData) + " / " + testData.data.length);
    };
    NeuralNetwork.prototype.updateMiniBatch = function (trainingData, miniBatchSize, eta) {
        var nabla_b = [];
        var nabla_w = [];
        var batch = trainingData.getBatch(miniBatchSize);
        if (batch.length <= 0)
            return;
        for (var b = 0; b < batch.length; ++b) {
            var res = this.backprop(batch[b].input, batch[b].output);
            var delta_nabla_b = res[0];
            var delta_nabla_w = res[1];
            for (var j = 0; j < delta_nabla_b.length; ++j) {
                if (b == 0)
                    nabla_b.push([]);
                for (var k = 0; k < delta_nabla_b[j].length; ++k) {
                    if (b == 0)
                        nabla_b[j].push(0);
                    nabla_b[j][k] += delta_nabla_b[j][k];
                }
            }
            for (var i = 0; i < delta_nabla_w.length; ++i) {
                if (b == 0)
                    nabla_w.push([]);
                for (var j = 0; j < delta_nabla_w[i].length; ++j) {
                    if (b == 0)
                        nabla_w[i].push([]);
                    for (var k = 0; k < delta_nabla_w[i][j].length; ++k) {
                        if (b == 0)
                            nabla_w[i][j].push(0);
                        nabla_w[i][j][k] += delta_nabla_w[i][j][k];
                    }
                }
            }
        }
        for (var l = 0; l < this.layers.length; ++l) {
            var weights = this.layers[l].getWeights();
            for (var i = 0; i < weights.length; ++i) {
                for (var j = 0; j < weights[i].length; ++j) {
                    var w = weights[i][j];
                    var nw = nabla_w[l][i][j];
                    weights[i][j] = w - (eta / batch.length) * nw;
                }
            }
            this.layers[l].putWeights(weights);
            var biases = this.layers[l].getBiases();
            for (var i = 0; i < biases.length; ++i) {
                var b = biases[i];
                var nb = nabla_b[l][i];
                biases[i] = b - (eta / batch.length) * nb;
            }
            this.layers[l].putBiases(biases);
        }
    };
    NeuralNetwork.prototype.backprop = function (input, output) {
        var nabla_b = [];
        var nabla_w = [];
        var activations = [];
        activations.push(input.slice());
        var zs = [];
        var activation = input;
        for (var i = 0; i < this.layers.length; ++i) {
            var z = this.layers[i].feedForwardRaw(activation);
            zs.push(z);
            activation = z.map(function (x) { return Util_1.Util.sigmoid(x); });
            activations.push(activation);
        }
        var delta = Util_1.Util.hadamard(this.costDerivative(activations[activations.length - 1], output), zs[zs.length - 1].map(function (x) { return Util_1.Util.sigmoidPrime(x); }));
        nabla_b.push([].slice.call(delta));
        nabla_w.push(this.costChangeWRTWeight(delta, activations[activations.length - 2]));
        for (var l = 2; l <= this.layers.length; ++l) {
            var z = zs[zs.length - l];
            var currentWeights = this.layers[this.layers.length - l + 1].getWeights();
            var dotResult = this.transposeMatrixDot(currentWeights, delta);
            delta = Util_1.Util.hadamard(dotResult, z.map(function (x) { return Util_1.Util.sigmoidPrime(x); }));
            nabla_b.push([].slice.call(delta.slice()));
            nabla_w.push(this.costChangeWRTWeight(delta, activations[activations.length - l - 1]));
        }
        nabla_b.reverse();
        nabla_w.reverse();
        return [nabla_b, nabla_w];
    };
    NeuralNetwork.prototype.costDerivative = function (outputActivations, testOutput) {
        if (outputActivations.length != testOutput.length)
            throw "NN.costDerivative: a and output different sizes";
        var result = outputActivations.slice();
        for (var i = 0; i < result.length; ++i) {
            result[i] -= testOutput[i];
        }
        return result;
    };
    NeuralNetwork.prototype.costChangeWRTWeight = function (delta, activation) {
        var dCdw = [];
        for (var j = 0; j < delta.length; ++j) {
            dCdw.push([]);
            for (var k = 0; k < activation.length; ++k) {
                dCdw[j].push(delta[j] * activation[k]);
            }
        }
        return dCdw;
    };
    NeuralNetwork.prototype.transposeMatrixDot = function (w, delta) {
        if (w.length != delta.length)
            throw "NN.transposeMatrixDot: w and delta different sizes";
        var result = [];
        for (var j = 0; j < w[0].length; ++j) {
            var d = 0;
            for (var i = 0; i < w.length; ++i) {
                d += w[i][j] * delta[i];
            }
            result.push(d);
        }
        return result;
    };
    return NeuralNetwork;
}());
exports.NeuralNetwork = NeuralNetwork;
