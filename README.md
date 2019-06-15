# PuzzleBox.NeuralNets

_This library is still in alpha._

**PuzzleBox.NeuralNets** is a [.NET library](https://www.nuget.org/packages/PuzzleBox.NeuralNets/0.1.0) for building and training neural networks.  It's designed mainly to be easy to use and extend.  The features are prioritised according to the needs of another project, for analysing the audio of bee hives.

It was inspired by the fluent interface of [ConvNetSharp](https://github.com/cbovar/ConvNetSharp), a port of [ConvNetJS](https://github.com/karpathy/convnetjs), but I wanted transposed convolutional layers which ConvNetSharp didn't support at the time.  I found other libraries overly difficult to learn and use, and I wanted a deeper understanding of neural networks which is why I decided to write this library.

```c#
// Build
var net = new Net(2)  // Input size 2
    .Dense(2)         // Layer output size 2
    .Sigmoid()
    .Dense(1)         // Layer output size 1
    .Sigmoid();

// Train
var trainer = new Trainer(net, 0.15f);
var cost = await trainer.TrainAsync(10000, trainingData);

// Run
var output = net.FeedForwards(input);
```

**Current features**

* Fluent interface
* Uses MathNet.Numerics which supports hardware acceleration.
* Layers
  * Dense
  * Convolution
  * Transpose Convolution
* Activation functions
  * Sigmoid
  * TanH
  * Relu
* Cost functions:
  * Squared error
  * Cross entropy
