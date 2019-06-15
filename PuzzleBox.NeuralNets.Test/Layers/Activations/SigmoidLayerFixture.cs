using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers.Activations
{
    public class SigmoidLayerFixture
    {
        [Test]
        public void should_feed_forward_correctly()
        {
            // Arrange
            var sigmoid = new Sigmoid(new Size(2, 2));

            var expected = new float[]
            {
                0.58f, 0.65f, 0.57f, 0.55f
            }.ToVector();

            // Act
            var actual = sigmoid.FeedForwards(new float[,]
            {
                { 0.31f, 0.27f },
                { 0.61f, 0.19f },
            });

            // Assert
            Assert.That(actual.Size.Dimensions.Length, Is.EqualTo(2));
            Assert.That(actual.Size.Dimensions[0], Is.EqualTo(2));
            Assert.That(actual.Size.Dimensions[1], Is.EqualTo(2));

            actual = (actual.ToMatrix() * 100).PointwiseRound() / 100;
            Assert.That(actual.Value, Is.EqualTo(expected));
        }

        [Test]
        public void should_back_propagate_correctly()
        {
            // Arrange
            var sigmoid = new Sigmoid(new Size(2, 2));

            var trainingRun = new TrainingRun(1)
            {
                Input = new float[]
                {
                    0.31f, 0.61f, 0.27f, 0.19f
                },
                OutputError = new float[]
                {
                    0.25f*0.61f + -0.15f*0.02f,
                    0.25f*0.96f + -0.15f*0.23f,
                    0.25f*0.82f + -0.15f*-0.50f,
                    0.25f*-1.00f + -0.15f*0.17f
                }
            };

            var expected = new float[,]
            {
                { 0.0364182f, 0.068628f },
                { 0.04675125f, -0.06818625f }
            };

            // Act
            sigmoid.BackPropagate(trainingRun);

            // Assert
            var actual = (trainingRun.InputError.ToMatrix() * 100).PointwiseRound() / 100;
            var expectedMatrix = (Matrix<float>.Build.DenseOfArray(expected) * 100).PointwiseRound() / 100;
            Assert.That(actual, Is.EqualTo(expectedMatrix));
        }
    }
}
