
import { NeuralNetwork } from '../';
import { prepareNN, QUADRANTS_DEPTH, colorFrom48Bit, positionToQuadrants } from './prepareNN';

var World = {
  imgWidth: 0,
  imgHeight: 0,
  img: 0,
  inCanvas: document.getElementById("fromCanvas"),
  outCanvas: document.getElementById("outCanvas"),
};
World.inCtx = World.inCanvas.getContext('2d');
World.outCtx = World.outCanvas.getContext('2d');
window.World = World;

$(main);
function main(){

  return loadImage().then(img => {
    World.img = img;
    World.imgWidth = img.width;
    World.imgHeight = img.height;

    World.outCanvas.width = World.imgWidth;
    World.outCanvas.height = World.imgHeight;
    World.inCanvas.width = World.imgWidth;
    World.inCanvas.height = World.imgHeight;
    World.inCtx.drawImage(World.img, 0, 0);

  });

}

window.startNN = function(){
  main().then(() => {

    $("#epochCanvases").text();

    console.log("Preparing neural network...");
    var { nn, trainingData, testData } = prepareNN(World);
    console.log("Running neural network...");

    run(nn, trainingData, testData);

    feedForwardImage(nn);
  });

}

function loadImage(){
  return new Promise((resolve, reject) => {
    var img = new Image();
    //img.src = 'static/house.jpg';
    img.src = $("#image").val() || 'static/cb2.jpg';
    img.onload = () => resolve(img);
  });
}


function run(nn, trainingData, testData) {

  console.log(trainingData, testData);

  var epochs = parseInt($("#epochs").val()); //40; //30; //10;
  var miniBatchSize = Math.floor(trainingData.data.length / 10);
  var eta = nn.learningRateEta;
  var printer = function(){ feedForwardImage(nn) }; // <--- draw between each epoch
  //var printer = () => { nn.learningRateEta--; };
  //var printer = () => { };
  nn.stochasticGradientDescent(trainingData, epochs, miniBatchSize, eta, printer, testData);

}

function feedForwardImage(nn){

  console.log("+ feedforward ---");
  var imgData = World.outCtx.getImageData(0, 0, World.imgWidth, World.imgHeight);

  for(var i=0; i < World.imgHeight; ++i){
    for(var j=0; j < World.imgWidth; ++j){
      var index = i * World.imgWidth + j;

      var qx = positionToQuadrants(i, World.imgHeight, QUADRANTS_DEPTH);
      var qy = positionToQuadrants(j, World.imgWidth, QUADRANTS_DEPTH);

      var out = nn.feedForward([...qx, ...qy]);
      var {r, g, b} = colorFrom48Bit(out);

      imgData.data[4 * index + 0] = r;
      imgData.data[4 * index + 1] = g;
      imgData.data[4 * index + 2] = b;
      imgData.data[4 * index + 3] = 255;
    }
  }
  console.log("- feedforward --- drawing");
  World.outCtx.putImageData(imgData, 0, 0);

  var newCanvas = $("<canvas></canvas>");
  $("#epochCanvases").append(newCanvas);
  var newCtx = newCanvas[0].getContext('2d');
  newCanvas[0].width = World.outCtx.canvas.width / 2;
  newCanvas[0].height = World.outCtx.canvas.height / 2;
  newCtx.drawImage(World.outCtx.canvas, 0, 0, newCanvas[0].width, newCanvas[0].height);
}
