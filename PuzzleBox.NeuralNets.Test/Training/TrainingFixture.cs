using System;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Test.TestHelpers;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test
{
    public class TrainingFixture
    {
        private const float AcceptableErrorThreshold = 1;

        [SetUp]
        public void SetUp()
        {
            MathNetExtensions.SetRandomSeed(2);
        }

        [Test]
        public async Task should_learn_NOT()
        {
            // Arrange
            var net = new Net(1)
                .Dense(1)
                .Sigmoid();
            var sut = new Trainer(net, 0.15f).UseCrossEntropyCost();

            // Act
            var finalCost = await sut.TrainAsync(300, TrainingData.NOT);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in TrainingData.NOT)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Assert.That(activation, Is.EqualTo((int)data.output));
            }
        }

        [Test]
        public async Task should_learn_AND()
        {
            // Arrange
            var net = new Net(2)
                .Dense(1)
                .Sigmoid();
            var sut = new Trainer(net, 0.15f).UseCrossEntropyCost();

            // Act
            var finalCost = await sut.TrainAsync(300, TrainingData.AND);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in TrainingData.AND)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Assert.That(activation, Is.EqualTo((int)data.output));
            }
        }

        [Test]
        public async Task should_learn_OR()
        {
            // Arrange
            var net = new Net(2)
                .Dense(1)
                .Sigmoid();
            var sut = new Trainer(net, 0.15f).UseCrossEntropyCost();

            // Act
            var finalCost = await sut.TrainAsync(500, TrainingData.OR);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in TrainingData.OR)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Assert.That(activation, Is.EqualTo((int)data.output));
            }
        }

        [Test]
        public async Task should_learn_XNOR()
        {
            // Arrange
            var net = new Net(2)
                .Dense(2)
                .Sigmoid()
                .Dense(1)
                .Sigmoid();
            var sut = new Trainer(net, 0.15f).UseCrossEntropyCost();

            // Act
            var finalCost = await sut.TrainAsync(20000, TrainingData.XNOR);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in TrainingData.XNOR)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Assert.That(activation, Is.EqualTo((int)data.output));
            }
        }

        [Test]
        public async Task should_learn_2d_features()
        {
            // Arrange
            var net = new Net(5, 5)
                .Convolution(new[] { 3, 3 }, 2)
                .Sigmoid()
                .Dense(1)
                .Sigmoid();
            var sut = new Trainer(net, 0.3f).UseCrossEntropyCost();

            // Act
            var finalCost = await sut.TrainAsync(20000, TrainingData.Lines2dPadded);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));
            Console.Out.WriteLine($"Cost: {finalCost}");

            foreach (var data in TrainingData.Lines2dPadded)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Console.Out.WriteLine($"Output: {activation}, Expected: {data.output.Value[0]}");
            }

            foreach (var data in TrainingData.Lines2dPadded)
            {
                var output = net.FeedForwards(data.input);
                var activation = (int)Math.Round(output, 0);
                Assert.That(activation, Is.EqualTo((int)data.output));
            }
        }

        [Test]
        public async Task should_learn_transposed_convolution()
        {
            // Arrange
            var net = new Net(1, 1)
                .ConvolutionTranspose(new Size(new[] { 2, 2 }))
                .Sigmoid();
            var sut = new Trainer(net, 0.01f).UseCrossEntropyCost();

            var trainingData = new (Tensor input, Tensor output)[]
            {
                (
                    new float[,] { { -1 } },
                    new float[,]
                    {
                        { 0, 1 },
                        { 1, 0 },
                    }
                ),
                (
                    new float[,] { { 1 } },
                    new float[,]
                    {
                        { 1, 0 },
                        { 0, 1 },
                    }
                ),
            };


            // Act
            var finalCost = await sut.TrainAsync(2000, trainingData);

            // Assert
            //Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));
            Console.Out.WriteLine($"Cost: {finalCost}");

            foreach (var data in trainingData)
            {
                var output = net.FeedForwards(data.input);
                Console.WriteLine("ACTUAL:" + output.Value.Map(x => Math.Round(x, 2)));
                Console.WriteLine("DESIRED: " + data.output.Value);
            }

            foreach (var data in trainingData)
            {
                var output = net.FeedForwards(data.input);
                Assert.That(output.Value.AlmostEqual(data.output.Value, 0.2f), Is.True);
            }
        }

        [Test]
        public async Task should_learn_transposed_convolution2()
        {
            // Arrange
            var net = new Net(1, 1)
                .ConvolutionTranspose(new Size(new[] { 4, 4 }))
                .Sigmoid();
            var sut = new Trainer(net, 0.10f).UseCrossEntropyCost();

            var trainingData = new (Tensor input, Tensor output)[]
            {
                (
                    new float[,] { { -1 } },
                    new float[,]
                    {
                        { 1, 1, 1, 1 },
                        { 1, 1, 1, 1 },
                        { 0, 0, 0, 0 },
                        { 0, 0, 0, 0 },
                    }
                ),
                (
                    new float[,] { { 1 } },
                    new float[,]
                    {
                        { 0, 0, 0, 0 },
                        { 0, 0, 0, 0 },
                        { 1, 1, 1, 1 },
                        { 1, 1, 1, 1 },
                    }
                ),
            };


            // Act
            var finalCost = await sut.TrainAsync(2000, trainingData);

            // Assert
            //Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));
            Console.Out.WriteLine($"Cost: {finalCost}");

            foreach (var data in trainingData)
            {
                var output = net.FeedForwards(data.input);
                Console.WriteLine("ACTUAL:" + output.Value.Map(x => Math.Round(x, 0)));
                Console.WriteLine("DESIRED: " + data.output.Value);
            }

            foreach (var data in trainingData)
            {
                var output = net.FeedForwards(data.input);
                Assert.That(output.Value.AlmostEqual(data.output.Value, 0.2f), Is.True);
            }
        }
    }
}
