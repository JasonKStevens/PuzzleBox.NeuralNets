using MathNet.Numerics;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class ConvolutionalLayerFixture : LayerFixtureBase<ConvolutionalLayer>
    {
        [SetUp]
        public void Setup()
        {
            _sut = new ConvolutionalLayer(new Size(3, 3), new Size(2, 2), 2, 2);
        }

        [Test]
        public void should_feed_forward_correctly()
        {
            // Arrange
            var input = new float[,] {
                { 0.76f, -1.34f, 2.41f },
                { -0.42f, 1.76f, 0.24f },
                { -1.24f, 0.43f, 0.76f },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.63f, -0.23f },
                { -0.14f, 0.41f },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 1.5674f, -1.5465f },
                { -0.3195f, 1.305f }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }

        [Test]
        public void should_back_propagate_correctly()
        {
            // Arrange
            var input = new float[,] {
                { 0.76f, -1.34f, 2.41f },
                { -0.42f, 1.76f, 0.24f },
                { -1.24f, 0.43f, 0.76f },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.63f, -0.23f },
                { -0.14f, 0.41f },
            }.ToMatrix());

            var trainingRun = new TrainingRun(1)
            {
                Input = input,
                Output = _sut.FeedForwards(new Tensor(input)),
                OutputError = new float[,] {
                    { 0.24f, 0.78f },
                    { -0.31f, 0.45f }
                }
            };

            var expectedInputError = new float[,] {
                { 0.037422f, 0.692316f, -0.257738f },
                { 1.157121f, -0.042478f, 0.271237f },
                { -0.258986f, 0.643897f, 0.335503f },
            }.ToMatrix();

            var expectedWeightsDelta = new float[,] {
                { 0.0594f, 1.1206f },
                { 1.8499f, 0.8183f },
            }.ToMatrix();

            // Act
            _sut.BackPropagate(trainingRun);

            // Assert
            Assert.That(trainingRun.WeightsDelta.AlmostEqual(expectedWeightsDelta, 0.001f), Is.True);
            Assert.That(trainingRun.InputError.ToMatrix().AlmostEqual(expectedInputError, 0.001f), Is.True);
        }

        [Test]
        public void should_feed_forward_correctly_with_multiple_kernels()
        {
            // Arrange
            _sut = new ConvolutionalLayer(new Size(new [] { 3, 3 }, 2), new Size(new[] { 2, 2 }, 4), 2, 2);

            var input = new Tensor(
                new Size(new[] { 3, 3 }, 2),
                new float[] {
                    0.76f, -0.42f, -1.24f,
                    -1.34f, 1.76f, 0.43f,
                     2.41f, 0.24f, 0.76f,
                    0.76f, -0.42f, -1.24f,
                    -1.34f, 1.76f, 0.43f,
                     2.41f, 0.24f, 0.76f
                });

            _sut.SetWeights(new float[,] {
                { 0.63f, -0.23f, 0.63f, -0.23f, 0.63f, -0.23f, 0.63f, -0.23f },
                { -0.14f, 0.41f, -0.14f, 0.41f, -0.14f, 0.41f, -0.14f, 0.41f },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 1.5674f, -1.5465f, 1.5674f, -1.5465f, 1.5674f, -1.5465f, 1.5674f, -1.5465f },
                { -0.3195f, 1.305f, -0.3195f, 1.305f, -0.3195f, 1.305f, -0.3195f, 1.305f }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(input);

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }

        [Test]
        public void should_feed_forward_with_single_out()
        {
            // Arrange
            _sut = new ConvolutionalLayer(new Size(2, 2), new Size(1, 1), 2, 2);

            var input = new float[,] {
                { 1, 2 },
                { 3, 4 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 5, 6 },
                { 7, 8 },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 5*1 + 6*2 + 7*3 + 8*4 }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }

        [Test]
        public void should_feed_forward_reducing_kernels()
        {
            // Arrange
            _sut = new ConvolutionalLayer(new Size(new[] { 2, 2 }, 1), new Size(1, 1), 2, 2);

            var input = new Tensor(
                new Size(new[] { 2, 2 }, 1),
                new float[] {
                    1, 3, 2, 4,
                });

            _sut.SetWeights(new float[,] {
                { 5, 6, 5, 6 },
                { 7, 8, 7, 8 },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 5*1 + 6*2 + 7*3 + 8*4 }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(input);

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }
    }
}
