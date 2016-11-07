# image generating neural network

(Try it yourself at http://tablekat.net/fun/nnimage/ !!!)

... with Stochastic Gradient Descent.  
... with image related stuff!

Essentially take an encoded x/y coordinate, and train the neural network on an image to learn what the color should be at that point!

Neural network stuff: `src/nn`  
Using it to draw images: `src/image`  
Based on https://github.com/tablekat/simple-neural-network

![](http://i.imgur.com/IfRNgoH.png)

## Run with

`npm run build` or `npm run watch` in one tab, and `npm start` in another.

### Referred to
[This book thing](http://neuralnetworksanddeeplearning.com/chap1.html) to figure out the SGD related stuff.

## Examples
1 hidden layer, 7 neurons, 4? eta  
![](http://i.imgur.com/h1iIuw3.png)

3 hidden layer, 7 neurons, 4 eta???
![](http://i.imgur.com/WwurDVl.png)

1 hidden layer, 45 neurons, not sure eta
![](http://i.imgur.com/CxYugkU.png)

1 hidden layer, 45 neurons, 4 eta
![](http://i.imgur.com/t4quz9q.png)

1 hidden layer, 45 hidden neurons per layer, 12 eta
![](http://i.imgur.com/ukK7MPe.png)
(beautiful checkerboard pattern!!!)

1 hidden layer, 45 neurons per layer, 20 eta!!!
![](http://i.imgur.com/cFtYmDZ.png)
(!!!!!)

Same as above but with 40 epochs:
![](http://i.imgur.com/0mmMwSO.jpg)

Same as above but with different start image:
![](http://i.imgur.com/lxpAGUo.png)
