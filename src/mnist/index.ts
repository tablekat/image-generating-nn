
import * as path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import { TrainingData } from '../nn/TrainingData';
import { NumberImg, Data } from './data';

var dataPath = path.resolve(__dirname, "../../data/mnistout.csv");

console.log("Starting load data...");
Data.loadFile(dataPath)
  .then((data) => {
    console.log("Data loaded...");

    var nn = new NeuralNetwork({
      numInputs: data.width * data.height,
      numHiddenLayers: 1,
      neuronsPerHiddenLayer: 7,
      numOutputs: 10,
      networkEvaluationSuccessThreshold: 0.05, //0.01,
      learningRateEta: 2, // 3 is perfect immediately almost, 2 goes slowly but gets to a 100% success rate! (with 0.1 threshold)
    });
    var trainingData = new TrainingData();
    for(var i=0; i < data.imgs.length; ++i){
      trainingData.train([].slice.call(data.imgs[i].imgData), data.imgs[i].valToArr());
    }
    var testData = trainingData.split(0.2);

    run(nn, trainingData, testData);

  });


function run(nn: NeuralNetwork, trainingData: TrainingData, testData: TrainingData) {

  var epochs = 10;
  var miniBatchSize = Math.floor(trainingData.data.length / 10);
  var eta = nn.learningRateEta;
  var printer = () => {};
  nn.stochasticGradientDescent(trainingData, epochs, miniBatchSize, eta, printer, testData);

}
