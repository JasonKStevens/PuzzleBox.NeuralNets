using System;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Test.TestHelpers;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class DenseLayerFixture : LayerFixtureBase<DenseLayer>
    {
        [SetUp]
        public void Setup()
        {
            _sut = new DenseLayer(2, 1);
        }

        [Test]
        public void should_feed_forward_correctly()
        {
            // Arrange
            var layer = new DenseLayer(3, 2);

            layer.SetWeights(new float[2, 4]
            {
                { 0.61f, 0.82f, 0.96f, -1 },
                { 0.02f, -0.5f, 0.23f, 0.17f },
            });

            var expected = new float[2]
            {
                0.61f*1 + 0.82f*0.57f + 0.96f*0.65f + -1*0.55f,
                0.02f*1 + -0.5f*0.57f + 0.23f*0.65f + 0.17f*0.55f
            }.ToVector();

            // Act
            var actual = layer.FeedForwards(new float[3]
            {
                0.57f, 0.65f, 0.55f
            });

            // Assert
            actual = (actual.Value * 100).PointwiseRound() / 100;
            expected = (expected * 100).PointwiseRound() / 100;
            Assert.That(actual.Value, Is.EqualTo(expected));
        }

        [Test]
        public void should_back_propagate_correctly()
        {
            // Arrange
            var layer = new DenseLayer(3, 2);

            layer.SetWeights(new float[2, 4]
            {
                { 0.61f, 0.82f, 0.96f, -1 },
                { 0.02f, -0.5f, 0.23f, 0.17f },
            });

            var input = new float[3]
            {
                0.57f, 0.65f, 0.55f
            };

            var trainingRun = new TrainingRun(1)
            {
                Input = input,
                Output = layer.FeedForwards(input),
                OutputError = new float[2] { 0.25f, -0.68f }
            };

            var expected = new float[3]
            {
                0.82f*0.25f + -0.5f*-0.68f,
                0.96f*0.25f + 0.23f*-0.68f,
                -1*0.25f + 0.17f*-0.68f,
            }.ToVector();

            // Act
            layer.BackPropagate(trainingRun);

            // Assert
            var actual = (trainingRun.InputError.Value * 100).PointwiseRound() / 100;
            expected = (expected * 100).PointwiseRound() / 100;
            Assert.That(actual, Is.EqualTo(expected));
        }

        [TestCase(1, 0)]
        [TestCase(0, 1)]
        public void should_simulate_NOT_gate(float input, int output)
        {
            // Arrange
            _sut = new DenseLayer(1, 1);
            _sut.SetWeights(TestWeights.NOT);

            // Act
            var activation = _sut.FeedForwards(input);
            var actualOutput = (int)Math.Round(ActivationFn.Sigmoid(activation), 0);

            // Assert
            Assert.That(actualOutput, Is.EqualTo(output));
        }

        [TestCase(new float[] { 0, 0 }, 0)]
        [TestCase(new float[] { 0, 1 }, 0)]
        [TestCase(new float[] { 1, 0 }, 0)]
        [TestCase(new float[] { 1, 1 }, 1)]
        public void should_simulate_AND_gate(float[] input, int output)
        {
            // Arrange
            _sut.SetWeights(TestWeights.AND);

            // Act
            var activation = _sut.FeedForwards(input);
            var actualOutput = (int)Math.Round(ActivationFn.Sigmoid(activation), 0);

            // Assert
            Assert.That(actualOutput, Is.EqualTo(output));
        }

        [TestCase(new float[] { 0, 0 }, 0)]
        [TestCase(new float[] { 0, 1 }, 1)]
        [TestCase(new float[] { 1, 0 }, 1)]
        [TestCase(new float[] { 1, 1 }, 1)]
        public void should_simulate_OR_gate(float[] input, int output)
        {
            // Arrange
            _sut.SetWeights(TestWeights.OR);

            // Act
            var activation = _sut.FeedForwards(input);
            var actualOutput = (int)Math.Round(ActivationFn.Sigmoid(activation), 0);

            // Assert
            Assert.That(actualOutput, Is.EqualTo(output));
        }

        [TestCase(new float[] { 0, 0 }, 1)]
        [TestCase(new float[] { 0, 1 }, 0)]
        [TestCase(new float[] { 1, 0 }, 0)]
        [TestCase(new float[] { 1, 1 }, 1)]
        public void should_simulate_XNOR_gate(float[] input, int output)
        {
            // Act
            float[] activation;

            _sut.SetWeights(TestWeights.AND);
            activation = _sut.FeedForwards(input);
            var a1 = (int)Math.Round(ActivationFn.Sigmoid(activation[0]), 0);

            _sut.SetWeights(TestWeights.NOR);
            activation = _sut.FeedForwards(input);
            var a2 = (int)Math.Round(ActivationFn.Sigmoid(activation[0]), 0);

            _sut.SetWeights(TestWeights.OR);
            activation = _sut.FeedForwards(a1, a2);
            var actualOutput = (int)Math.Round(ActivationFn.Sigmoid(activation[0]), 0);

            // Assert
            Assert.That(actualOutput, Is.EqualTo(output));
        }
    }
}
