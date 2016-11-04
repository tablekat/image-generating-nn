
import { Util } from './Util'
import { Neuron } from './Neuron'
import { NeuronLayer } from './NeuronLayer'
import { TrainingData, TrainingPair } from './TrainingData'

export interface NeuralNetworkArgs {
  numInputs?: number;
  numHiddenLayers?: number;
  neuronsPerHiddenLayer?: number;
  numOutputs?: number;
  networkEvaluationSuccessThreshold?: number;
  learningRateEta?: number;
}

function def(thing) {
  return typeof thing !== "undefined";
}

export class NeuralNetwork implements NeuralNetworkArgs{

  layers: NeuronLayer[];

  /*** defaults ***/
  numInputs: number                           = 9;
  numHiddenLayers: number                     = 2;
  neuronsPerHiddenLayer: number               = 6;
  numOutputs: number                          = 5;
  networkEvaluationSuccessThreshold: number   = 0.01;
  learningRateEta: number                     = 0.01; // 3.0;

  constructor(args?: NeuralNetworkArgs){
    // http://neuralnetworksanddeeplearning.com/chap1.html
    if(!args) args = {};
    if(def(args.numInputs))        this.numInputs = args.numInputs;
    if(def(args.numHiddenLayers))  this.numHiddenLayers = args.numHiddenLayers;
    if(def(args.neuronsPerHiddenLayer))
                                   this.neuronsPerHiddenLayer = args.neuronsPerHiddenLayer;
    if(def(args.numOutputs))       this.numOutputs = args.numOutputs;
    if(def(args.networkEvaluationSuccessThreshold))
                                   this.networkEvaluationSuccessThreshold = args.networkEvaluationSuccessThreshold;
    if(def(args.learningRateEta))  this.learningRateEta = args.learningRateEta;

    this.layers = [];
    var inputs = this.numInputs;
    for(var i=0; i < this.numHiddenLayers; ++i){
      this.layers.push(new NeuronLayer(inputs, this.neuronsPerHiddenLayer));
      inputs = this.neuronsPerHiddenLayer;
    }
    this.layers.push(new NeuronLayer(inputs, this.numOutputs));

  }

  public feedForward(inputs: Float64Array): Float64Array{
    var outputs = inputs.slice();
    for(var i=0; i < this.layers.length; ++i){
      outputs = this.layers[i].feedForward(outputs);
    }
    return outputs;
  }

  public export(weights: Float64Array, offset: number): number{
    for(var i=0; i < this.numInputs; ++i){
      offset = this.layers[i].export(weights, offset);
    }
    return offset;
  }
  public import(weights: Float64Array, offset: number): number{
    for(var i=0; i < this.numInputs; ++i){
      offset = this.layers[i].import(weights, offset);
    }
    return offset;
  }

  public evaluate(testData: TrainingData, threshold?: number){
    threshold = threshold || this.networkEvaluationSuccessThreshold;


    var successes = 0;
    for(var i=0; i < testData.data.length; ++i){
      var result = this.evaluatePair(testData.data[i], threshold);
      if(result < 0) return result;
      successes += result;
    }

    return successes;
  }

  private evaluatePair(testPair: TrainingPair, threshold?: number){
    threshold = threshold || this.networkEvaluationSuccessThreshold;
    var output = this.feedForward(testPair.input);
    var amount = output.length;
    if(amount != testPair.output.length) return -1;

    for(var i=0; i < amount; ++i){
      var diff = testPair.output[i] - output[i];
      if(Math.abs(diff) > threshold) return 0;
    }

    return 1;
  }


  /***** STOCHASTIC GRADIENT DESCENT *****/

  public stochasticGradientDescent(trainingData: TrainingData, epochs: number,
      miniBatchSize: number, eta: number, printer?: Function, testData?: TrainingData){
    /* "Train the neural network using mini-batch stochastic gradient descent" */

    var doEvaluation = typeof(printer) == "function"
                          && (testData instanceof TrainingData)
                          && testData.data.length > 0;
    //trainingData.reset();

    var i = 0;
    var loop = () => {
      trainingData.reset();
      for(var j = 0; j < trainingData.data.length / miniBatchSize; ++j){
        this.updateMiniBatch(trainingData, miniBatchSize, eta);
      }

      //console.log(doEvaluation, printer, testData);
      if(doEvaluation){
        this.logEpochEvaluation(i, printer, testData);
      }

      i++;
      if(i < epochs) setTimeout(loop, 0);
    }
    setTimeout(loop, 0);

  }

  public logEpochEvaluation(epoch: number, printer?: Function, testData?: TrainingData){
    console.log("Epoch: " + epoch + ": " + this.evaluate(testData) + " / " + testData.data.length);
    printer("Epoch: " + epoch + ": " + this.evaluate(testData) + " / " + testData.data.length);
  }

  public updateMiniBatch(trainingData: TrainingData, miniBatchSize: number, eta: number){
     var nabla_b: number[][] = [];
     var nabla_w: number[][][] = [];

     var batch = trainingData.getBatch(miniBatchSize);
     if(batch.length <= 0) return;

     for(var b = 0; b < batch.length; ++b){
       var res = this.backprop(batch[b].input, batch[b].output);
       var delta_nabla_b = <number[][]>res[0];
       var delta_nabla_w = <number[][][]>res[1];

       // update nabla_b
       for(var j=0; j < delta_nabla_b.length; ++j){
         if(b == 0) nabla_b.push([]);
         for(var k=0; k < delta_nabla_b[j].length; ++k){
           if(b == 0) nabla_b[j].push(0);
           nabla_b[j][k] += delta_nabla_b[j][k];
         }
       }
       // update nabla_w
       for(var i=0; i < delta_nabla_w.length; ++i){
         if(b == 0) nabla_w.push([]);
         for(var j=0; j < delta_nabla_w[i].length; ++j){
           if(b == 0) nabla_w[i].push([]);
           for(var k=0; k < delta_nabla_w[i][j].length; ++k){
             if(b == 0) nabla_w[i][j].push(0);
             nabla_w[i][j][k] += delta_nabla_w[i][j][k];
           }
         }
       }
     }

     // now update the weights and biases!!
     for(var l = 0; l < this.layers.length; ++l){
       // update weights
       var weights = this.layers[l].getWeights();
       for(var i=0; i < weights.length; ++i){
         for(var j=0; j < weights[i].length; ++j){
           //console.log(l, i, j, weights, nabla_w[l]);
           var w = weights[i][j];
           var nw = nabla_w[l][i][j];
           weights[i][j] = w - (eta / batch.length) * nw;
         }
       }
       this.layers[l].putWeights(weights);
       // update biases!
       var biases = this.layers[l].getBiases();
       for(var i=0; i < biases.length; ++i){
         var b = biases[i];
         var nb = nabla_b[l][i];
         biases[i] = b - (eta / batch.length) * nb;
       }
       this.layers[l].putBiases(biases);
     }
  }

  public backprop(input: Float64Array, output: Float64Array){

    var nabla_b: number[][] = [];
    var nabla_w: number[][][] = [];

    // Manual feedforward
    var activations: Float64Array[] = []; // :: [[Double]]
    activations.push(input.slice());
    var zs: Float64Array[] = []; // :: [[Double]]
    var activation = input;

    for(var i=0; i < this.layers.length; ++i){
      var z = this.layers[i].feedForwardRaw(activation);
      zs.push(z);
      activation = z.map(x => Util.sigmoid(x));
      activations.push(activation);
    }

    // Backward pass!
    var delta = Util.hadamard(
      this.costDerivative(activations[activations.length - 1], output),
      zs[zs.length - 1].map(x => Util.sigmoidPrime(x))); // :: [Double]
    nabla_b.push([].slice.call(delta));
    //nabla_w.push(NNUtils.dot(delta, activations[activations.length - 2]));
    nabla_w.push(this.costChangeWRTWeight(delta, activations[activations.length - 2]));

    for(var l = 2; l <= this.layers.length; ++l){
      var z = zs[zs.length - l];

      var currentWeights = this.layers[this.layers.length - l + 1].getWeights(); // :: [[Double]]
      var dotResult = this.transposeMatrixDot(currentWeights, delta);// :: [Double]

      delta = Util.hadamard(dotResult, z.map(x => Util.sigmoidPrime(x)));
      nabla_b.push([].slice.call(delta.slice()));
      nabla_w.push(this.costChangeWRTWeight(delta, activations[activations.length - l - 1]));
    }

    nabla_b.reverse();
    nabla_w.reverse();
    return [nabla_b, nabla_w];
  }

  public costDerivative(outputActivations: Float64Array, testOutput: Float64Array){
    if(outputActivations.length != testOutput.length) throw "NN.costDerivative: a and output different sizes";
    var result = outputActivations.slice();
    for(var i=0; i < result.length; ++i){
      result[i] -= testOutput[i];
    }
    return result;
  }

  public costChangeWRTWeight(delta: Float64Array, activation: Float64Array){
    var dCdw: number[][] = []; // :: [[Double]]
    // Neuron j, weight k
    // (BP4) dC/dw_(ljk) = a_((l-1),k) * delta_(lj)
    for(var j=0; j < delta.length; ++j){
      dCdw.push([]);
      for(var k = 0; k < activation.length; ++k){
        dCdw[j].push(delta[j] * activation[k]);
      }
    }
    return dCdw;
  }

  public transposeMatrixDot(w: Float64Array[], delta: Float64Array){
    if(w.length != delta.length) throw "NN.transposeMatrixDot: w and delta different sizes";
    var result: number[] = []; // :: [Double]
    for(var j=0; j < w[0].length; ++j){
      var d = 0;
      for(var i=0; i < w.length; ++i){
        d += w[i][j] * delta[i];
      }
      result.push(d);
    }
    return result;
  }
}
