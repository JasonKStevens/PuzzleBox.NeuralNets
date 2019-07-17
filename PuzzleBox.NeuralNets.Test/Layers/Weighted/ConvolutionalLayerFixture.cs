using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using System.Collections;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class ConvolutionalLayerFixture
    {
        private ConvolutionalLayer _sut;

        // TODO: Test for kernelCount

        [TestCase(1, 1, 1)]
        [TestCase(2, 1, 2)]
        [TestCase(2, 2, 1)]
        [TestCase(3, 1, 3)]
        [TestCase(3, 3, 3)]
        [TestCase(4, 1, 4)]
        [TestCase(4, 2, 3)]
        [TestCase(4, 4, 3)]
        public void should_output_tensor_of_expected_size(int inputSize, int weightLength, int expectedOutputSize)
        {
            // Arrange
            _sut = (ConvolutionalLayer)new Net(inputSize)
                .Convolution(new [] { weightLength, 1 }, kernelCount: 1, strideArray: new [] { 1 })
                .Layers[0];

            var input = new float[inputSize];

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.Size.Dimensions.Length, Is.EqualTo(1));
            Assert.That(output.Size.Dimensions[0], Is.EqualTo(expectedOutputSize));
        }
    }
}
