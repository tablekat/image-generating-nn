
import { NeuralNetwork, TrainingData } from '../';

export const QUADRANTS_DEPTH = 8; // Split into quadrants 8 times.
export const BIT_COLORS = false;

export function prepareNN(World) {
  var nn = new NeuralNetwork({
    numInputs: 2 * QUADRANTS_DEPTH, // x, y position in quadrant space
    numHiddenLayers: 1, //1,
    neuronsPerHiddenLayer: 45, //7,
    numOutputs: BIT_COLORS ? 16 * 3 : 3, // r, g, b, each with 16 values 00 - ff
    networkEvaluationSuccessThreshold: 0.1, //0.65, //0.01,
    learningRateEta: 18, //4, //0.05, //0.4,
  });
  /*var trainingData = new TrainingData();
  for(var i=0; i < data.imgs.length; ++i){
    trainingData.train([].slice.call(data.imgs[i].imgData), data.imgs[i].valToArr());
  }*/
  var trainingData = getTrainingData(World);
  var testData = trainingData.split(0.2);

  return { nn, trainingData, testData };
}

function getTrainingData(World){
  var imgData = World.inCtx.getImageData(0, 0, World.imgWidth, World.imgHeight);

  var trainingData = new TrainingData();

  var HeightLimit = World.imgHeight; //World.imgHeight; // 50, 158
  var WidthLimit = World.imgWidth; //World.imgWidth; // 50

  for(var i=0; i < HeightLimit; ++i){
    for(var j=0; j < WidthLimit; ++j){
      var index = i * World.imgWidth + j;
      var r = imgData.data[4 * index + 0];
      var g = imgData.data[4 * index + 1];
      var b = imgData.data[4 * index + 2];
      var qx = positionToQuadrants(i, World.imgHeight, QUADRANTS_DEPTH);
      var qy = positionToQuadrants(j, World.imgWidth, QUADRANTS_DEPTH);
      //console.log([index, i, j], '=>', colorTo48Bit(r, g, b))
      trainingData.train([...qx, ...qy], colorTo48Bit(r, g, b));
    }
  }

  return trainingData;
}

export function colorTo48Bit(r, g, b){
  if(!BIT_COLORS){
    return [r / 255, g / 255, b / 255];
  }
  var r = Math.floor(r / 16);
  var g = Math.floor(g / 16);
  var b = Math.floor(b / 16);
  var out = [];
  // Make the output bitwise yes or no for if the color is each value.
  for(var i=0; i < 16; ++i) out[i +  0] = 1 * (i == r);
  for(var i=0; i < 16; ++i) out[i + 16] = 1 * (i == g);
  for(var i=0; i < 16; ++i) out[i + 32] = 1 * (i == b);
  return out;
}

export function colorFrom48Bit(arr){
  if(!BIT_COLORS){
    return { r: arr[0] * 255, g: arr[1] * 255, b: arr[2] * 255};
  }
  var rs = arr.slice(0, 16);
  var gs = arr.slice(16, 32);
  var bs = arr.slice(32, 48);
  var r = rs.indexOf(Math.max.apply(null, rs));
  var g = gs.indexOf(Math.max.apply(null, gs));
  var b = bs.indexOf(Math.max.apply(null, bs));
  return { r: 16 * r, g: 16 * g, b: 16 * b};
}

export function positionToQuadrants(x, maxX, depth){
  var out = [];
  for(var i=0; i < depth; ++i){
    if(x < maxX / 2){
      out[i] = 0; // 0 = left half of each sub half
      maxX = maxX / 2;
    }else{
      out[i] = 1;
      x -= maxX / 2;
      maxX = maxX / 2;
    }
  }
  return out;
}
window.positionToQuadrants = positionToQuadrants;
