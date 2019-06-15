using System;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Test.TestHelpers;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test
{
    public class AutoencodingFixture
    {
        private const float AcceptableErrorThreshold = 1;

        [SetUp]
        public void SetUp()
        {
            MathNetExtensions.SetRandomSeed(0);
        }

        [Test]
        public async Task should_learn_AND_autoencoding()
        {
            // Arrange
            var net = new Net(2)
                .Dense(1)
                .Sigmoid()
                .Dense(2)
                .Sigmoid()
                .Dense(2)
                .Sigmoid();

            var sut = new Trainer(net, 0.15f);
            var authoEncodingData = TrainingData.AND.Select(i => (input: i.input, output: i.input));

            // Act
            var finalCost = await sut.TrainAsync(100000, authoEncodingData);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Console.WriteLine("ACTUAL:" + output.Value);
                Console.WriteLine("DESIRED: " + data.output.Value);
            }

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Assert.That(output.Value.AlmostEqual(data.output.Value, 0.2f), Is.True);
            }
        }

        [Test]
        public async Task should_learn_OR_autoencoding()
        {
            // Arrange
            var net = new Net(2)
                .Dense(1)
                .Sigmoid()
                .Dense(2)
                .Sigmoid()
                .Dense(2)
                .Sigmoid();

            var sut = new Trainer(net, 0.15f);
            var authoEncodingData = TrainingData.OR.Select(i => (input: i.input, output: i.input));

            // Act
            var finalCost = await sut.TrainAsync(100000, authoEncodingData);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Console.WriteLine("ACTUAL:" + output.Value);
                Console.WriteLine("DESIRED: " + data.output.Value);
            }

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Assert.That(output.Value.AlmostEqual(data.output.Value, 0.2f), Is.True);
            }
        }

        [Test]
        public async Task should_learn_simple_2d_features()
        {
            // Arrange
            var net = new Net(5, 5)
                .Convolution(new Size(new [] { 2, 2 }, 2), 3, 3)
                .Relu()
                .Dense(new Size(new [] { 1, 1 }))
                .Relu()
                .ConvolutionTranspose(new Size(new[] { 5, 5 }, 1))
                .Relu();
            var sut = new Trainer(net, 0.10f).UseCrossEntropyCost();
            var authoEncodingData = TrainingData.Lines2dPadded.Select(i => (input: i.input, output: i.input));

            // Act
            var finalCost = await sut.TrainAsync(100000, authoEncodingData);

            // Assert
            Assert.That(finalCost, Is.LessThan(AcceptableErrorThreshold));
            Console.Out.WriteLine($"Cost: {finalCost}");

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Console.WriteLine("ACTUAL:" + output.Value.Map(x => Math.Round(x, 0)));
                Console.WriteLine("DESIRED: " + data.output.Value);
            }

            foreach (var data in authoEncodingData)
            {
                var output = net.FeedForwards(data.input);
                Assert.That(output.Value.AlmostEqual(data.output.Value, 0.2f), Is.True);
            }
        }
    }
}
