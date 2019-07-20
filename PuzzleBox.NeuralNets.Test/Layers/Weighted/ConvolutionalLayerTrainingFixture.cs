using MathNet.Numerics;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class ConvolutionalLayerTrainingFixture : LayerFixtureBase<ConvolutionalLayer>
    {
        [SetUp]
        public void Setup()
        {
            _sut = new ConvolutionalLayer(
                inputSize: new Size(3, 1),
                outputSize: new Size(2, 1),
                paddingArray: new [] { 0, 0 },
                strideArray: new[] { 1, 1 },
                weightLength: new[] { 2, 1 });
        }

        [Test]
        public void should_feed_forward_correctly()
        {
            // Arrange
            var input = new float[,] {
                { 1, 2, 3 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.2f, 0.7f },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 1.6f, 2.5f },
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
                { 1, 2, 3 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.2f, 0.7f },
            }.ToMatrix());

            var trainingRun = new TrainingRun(1)
            {
                Input = input,
                Output = _sut.FeedForwards(new Tensor(input)),
                OutputError = new float[,] {
                    { 0.2f, -0.5f }
                }
            };

            var expectedInputError = new float[,] {
                { 0.04f, 0.04f, -0.35f },
            }.ToMatrix();

            var expectedWeightsDelta = new float[,] {
                { -0.8f, -1.1f },
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
            _sut = new ConvolutionalLayer(
                inputSize: new Size(new[] { 3, 3 }, 2),
                outputSize: new Size(new[] { 2, 2 }, 4),
                paddingArray: new[] { 0, 0 },
                strideArray: new[] { 1, 1 },
                weightLength: new[] { 2, 2 });

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
            _sut = new ConvolutionalLayer(
                inputSize: new Size(2, 2),
                outputSize: new Size(1, 1),
                paddingArray: new[] { 0, 0 },
                strideArray: new[] { 1, 1 },
                weightLength: new[] { 2, 2 });

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
            _sut = new ConvolutionalLayer(
                inputSize: new Size(new[] { 2, 2 }, 2),
                outputSize: new Size(1, 1),
                paddingArray: new[] { 0, 0 },
                strideArray: new[] { 1, 1 },
                weightLength: new[] { 2, 2 });

            var input = new Tensor(
                new Size(new[] { 2, 2 }, 2),
                new float[] {
                    1, 3, 2, 4, 5, 7, 6, 8
                });

            _sut.SetWeights(new float[,] {
                { 5, 6 },
                { 7, 8 },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { (5*1 + 6*2 + 7*3 + 8*4 + 5*5 + 6*6 + 7*7 + 8*8) / 2 }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(input);

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }
    }
}
