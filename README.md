# PuzzleBox.NeuralNets

__This library is still in alpha__

**PuzzleBox.NeuralNets** is a [.NET library](https://www.nuget.org/packages/PuzzleBox.NeuralNets/0.1.0) for building and training neural networks.

The focus is ease of use:

```c#
// Build
var net = new Net(2)
    .Dense(2)
    .Sigmoid()
    .Dense(1)
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
