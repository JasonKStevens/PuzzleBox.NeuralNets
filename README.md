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

![XOR](https://lh3.googleusercontent.com/5DsvC9cpPtmMQYwlkaPV9__go0c193v1k-VDuAXh6OSeSHA6HRQuw4xwU5_dke6zDNOsEolhVlHIvyPb5UmNxat08S9nLEXOcslVS14KKMjKQzBqXC7Q4yfpmcgPvOfBUkmufC7T9pZ2UlUSilgeEjiTsiX2xozKlEHwizEONYcD7pinv7AmjuOqMsrWMCpuu-8f9LPLpB5dmvcMssPYWInSAgckSeGignUQ12n5xERMEDw11syw-htqTJbdMZz841IlDDW3VkaJJ-AisLnBU7CYG2UVRGOMKzPxT529TvFbt9fS1VPkuhbManlt8gKNspQtxDgmWL7BcOQGTh5lFpp5XI5CYAC7prmF0t8W1oBG7NwnwEXc4Eu_Dfg2K6MrMZUvXPEzwBjm_mFMzu2rWF5Yiwnn-xMWE6CKrcnsxhaPZXIPKmEwAS40YlM_ClC7yajuu8_qR8QCZme5h4M-HToFnfDu0wF7Ad1dsYEUKVbgm0FXOcRRr_lFDwPdY-fB66uN0VKdSxVIxstXpImc4ppEoZAcy-kT0pVMZkQkymtYzKYf6IDzSpT-RAhIbsMaT0CBJ9590pIJC6c0c6f7vpJXEzya7wsRg8FnWwcqSu84gIVbVVuhkeMnqFMoybs_Yr7gxD2xIRTl2cDx99B8MhG7BefIh2Bf=w319-h294-no)

***More Complex Example***

This example is more involved than a truth table with values between 0 and 1 forming a circular shape.

![More Complex Example](https://lh3.googleusercontent.com/XCFUf8sDZ_aygefG146KWbWwiYJKRp40vOk2w84RrwWlw54VAgQ87PgpRtAuvJurapRsGWij8IAoZOXTfrmlfgyRRSYGGSnt5O9dlDgL4VLZmW4YagBjXOqtkTXxNpmzVSgNsXOpu7sQDRLkksmMRtz5fDFOk5ETkmInByrKFnH6YD2x_L_6QeC1zUIXFXshdlL96CRqD5wtVzRiN8r_lTsPq4GdGyX-f5zLpAJNMY4085wJw92KSnB7PZ4tYj7vLvzxbJL_hVVIX8MqbKL4d-uzmb3eFUEfIRnzTfQYNepsH36G7iMHvxzjOLoxxf3Vp5WAvLK1XEp9X-4lUHGWuajIbUcapuAfyxi8YK-qzyylb6y9RhRhr22Sw6ETHp4MBQvuVM5rOVkBPqcwHKe7SzmaIXQqZDr_5uqnKixXVprqO8yjPxCbVLo3Mqfq9HypyAlivP2_m5LNC5ic-6Qsczic5s143-Z2Gv7ItkX9A8G7ai7fgavd3H-u2d5NasqGv65Xs_lqT5UtB4aTZwXkzVwSbqZGqXsOV7b4M3jESM-A80MgJ3RXUmq6pEmlOaVVu_f96YbWZvtMvH9OUWsdNDfBo9c0jazBIfeaOMb9FGChYv2TR6iSdR_97-N_EDc0RcniQiEPXq4UuvAdjl5CarGKYvjGXMVg=w319-h294-no)

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
