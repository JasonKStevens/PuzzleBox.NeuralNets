# PuzzleBox.NeuralNets

[![Build Status](https://dev.azure.com/jasonkstevens/PuzzleBox/_apis/build/status/PuzzleBox.NeuralNets?branchName=master)](https://dev.azure.com/jasonkstevens/PuzzleBox/_build/latest?definitionId=1&branchName=master)

_This library is in beta._

**PuzzleBox.NeuralNets** is a [.NET library](https://www.nuget.org/packages/PuzzleBox.NeuralNets/0.1.0) for building and training neural networks.  It's designed mainly to be easy to use and extend.  The features are prioritised according to the needs of another project, for analysing the audio of bee hives.

It was inspired by the fluent interface of [ConvNetSharp](https://github.com/cbovar/ConvNetSharp), a port of [ConvNetJS](https://github.com/karpathy/convnetjs), but I wanted transposed convolutional layers which ConvNetSharp didn't support at the time.  I found other libraries overly difficult to learn and use, and I wanted a deeper understanding of neural networks which is why I decided to write this library.

**Examples**

***XOR***
```c#
// Build
var net = new Net(2)  // Input size 2
    .Dense(2)         // Fully-connected layer with output size 2
    .Sigmoid()
    .Dense(1)         // Fully-connected layer with output size 1
    .Sigmoid();

// Train
var trainingData = new (Tensor input, Tensor output)[]
    {
        (new float[] { 0, 0 }, 0),
        (new float[] { 0, 1 }, 1),
        (new float[] { 1, 0 }, 1),
        (new float[] { 1, 1 }, 0),
    };

var trainer = new Trainer(net, 0.15f);
var cost = await trainer.TrainAsync(10000, trainingData);

// Run
var output = net.FeedForwards(input);
```

![XOR](https://lh3.googleusercontent.com/pw/ACtC-3eb4eGFXRjCvLaCyoB_YzG5bnXOVuERn2qRUaH_KDeM7QnnEE4uaiUh7hToGZiBPmYA6uNWVtN4DodrGaW8G4tD7sSU4F-EyRlO2yfrljzjmA_WHELz_RlM-apDUdnx7obL7SUutiMgbxdy_090wxwVlQ=w1000-no-tmp.jpg)

***More Complex Example***

This example is more involved than a truth table with values between 0 and 1 forming a circular shape.

![More Complex Example](https://lh3.googleusercontent.com/pw/ACtC-3dvcIG1kLjy29GXZYlsEeWfg4PTi7Viwb8n-0qqoR3KX9UjNigRGa5y2ZMEcANM7ZBca4Sl7JJK-LUFeYXd79GKJGbmHmSM0CUfNmkhAwHAgARGY-mYyWHIbd3N_o7uLQmPcAy2rfvcKuyE5iHpqK07Tg=w1000-no-tmp.jpg)

***CNN Autoencoder***
```c#
var net = new Net(5, 5)                        // Input size 5x5
    .Convolution(new [] { 3, 3 }, 2)           // Two 3x3 kernels
    .Relu()
    .Dense(new Size(new [] { 1, 1 }))          // Compress input to single value
    .Relu()
    .ConvolutionTranspose(new[] { 5, 5 }, 2)   // Two 5x5 kernels
    .Sigmoid()
    .Dense(new Size(new[] { 5, 5 }))
    .Sigmoid();
```

**Current Features**

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

**Possible Features**

* Layers
  * CNN with bias
  * N-dimensional CNN
  * Transformer
