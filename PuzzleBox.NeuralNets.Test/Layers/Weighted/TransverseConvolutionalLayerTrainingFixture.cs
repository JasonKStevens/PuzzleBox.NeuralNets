using MathNet.Numerics;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class TransverseConvolutionalLayerTrainingFixture : LayerFixtureBase<ConvolutionalLayer>
    {
        [SetUp]
        public void Setup()
        {
            _sut = new ConvolutionalLayer(
                inputSize: new Size(1, 1),
                outputSize: new Size(2, 2),
                weightLength: new [] { 2, 2 },
                strideArray: new[] { 1, 1 },
                paddingArray: new[] { 1, 1 });
        }

        [Test]
        public void should_feed_forward_correctly()
        {
            // Arrange
            var input = new float[,] {
                { 3f },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 1f, 2f },
                { 3f, 4f },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 1*3, 2*3 },
                { 3*3, 4*3 }
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }

        [Test]
        public void should_feed_forward_correctly_v2()
        {
            // Arrange
            _sut = new ConvolutionalLayer(
                inputSize: new Size(4, 4),
                outputSize: new Size(6, 6),
                weightLength: new[] { 3, 3 },
                strideArray: new[] { 1, 1 },
                paddingArray: new[] { 2, 2 });

            var input = new float[,] {
                { 1, 3, 2, 1 },
                { 1, 3, 3, 1 },
                { 2, 1, 1, 3 },
                { 3, 2, 3, 3 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 1, 2, 3 },
                { 0, 1, 0 },
                { 2, 1, 2 },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 1, 5, 11, 14, 8, 3 },
                { 1, 6, 15, 18, 12, 3 },
                { 4, 13, 21, 21, 15, 11 },
                { 5, 17, 28, 27, 25, 11 },
                { 4, 7, 9, 12, 8, 6 },
                { 6, 7, 14, 13, 9, 6 },
            }.ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.ToMatrix().AlmostEqual(expectedOutput, 0.001f), Is.True);
        }

        [Test]
        public void should_feed_forward_correctly_v3()
        {
            // Arrange
            var input = new float[,] {
                { 1 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.2f, 0.5f },
                { 0.3f, 0.4f },
            }.ToMatrix());

            var expectedOutput = new float[,] {
                { 0.2f, 0.5f },
                { 0.3f, 0.4f },
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
                { 1 },
            }.ToMatrix();

            _sut.SetWeights(new float[,] {
                { 0.2f, 0.5f },
                { 0.3f, 0.4f },
            }.ToMatrix());

            var trainingRun = new TrainingRun(1)
            {
                Input = input,
                Output = _sut.FeedForwards(new Tensor(input)),
                OutputError = new float[,] {
                    { 0.1f, 0 },
                    { 0, 0 }
                }
            };

            var expectedInputError = new float[,] {
                { 0.04f },
            }.ToMatrix();

            var expectedWeightsDelta = new float[,] {
                { 0.1f, 0 },
                { 0, 0 },
            }.ToMatrix();

            // Check application of delta fixes error
            _sut.SetWeights(new float[,] {
                { 0.3f, 0.5f },
                { 0.3f, 0.4f },
            }.ToMatrix());

            var expectedNewOutput = new float[,] {
                { 0.3f, 0.5f },
                { 0.3f, 0.4f },
            }.ToMatrix();

            var newOutput = _sut.FeedForwards(new Tensor(input));

            // Act
            _sut.BackPropagate(trainingRun);

            // Assert
            Assert.That(trainingRun.WeightsDelta.AlmostEqual(expectedWeightsDelta, 0.001f), Is.True);
            Assert.That(trainingRun.InputError.ToMatrix().AlmostEqual(expectedInputError, 0.001f), Is.True);

            Assert.That(newOutput.ToMatrix().AlmostEqual(expectedNewOutput, 0.001f), Is.True);
        }
    }
}
