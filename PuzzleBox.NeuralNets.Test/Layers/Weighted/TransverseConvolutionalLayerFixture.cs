using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using System;
using System.Collections;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class TransverseConvolutionalLayerFixture
    {
        private ConvolutionalLayer _sut;

        [Test, TestCaseSource("TestCases")]
        public void should_output_tensor_of_correct_size(int inputSize, int outputSize)
        {
            // Arrange
            _sut = (ConvolutionalLayer)new Net(inputSize, inputSize)
                .ConvolutionTranspose(new Size(new[] { outputSize, outputSize }))
                .Layers[0];

            var input = new float[inputSize, inputSize].ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.Size.Dimensions.Length, Is.EqualTo(2));
            Assert.That(output.Size.Dimensions[0], Is.EqualTo(outputSize));
            Assert.That(output.Size.Dimensions[1], Is.EqualTo(outputSize));
        }

        [TestCase(2, 1)]
        [TestCase(3, 2)]
        public void should_throw_for_input_sizes_greater_than_output(int inputSize, int outputSize)
        {
            Assert.Throws<ArgumentException>(() =>
                _sut = (ConvolutionalLayer)new Net(inputSize, inputSize)
                    .ConvolutionTranspose(new Size(new[] { outputSize, outputSize }))
                    .Layers[0]);
        }

        [TestCase(1, 0)]
        [TestCase(0, 1)]
        [TestCase(-1, 1)]
        [TestCase(1, -1)]
        public void should_throw_for_sizes_less_than_one(int inputSize, int outputSize)
        {
            Assert.Throws<ArgumentException>(() =>
                _sut = (ConvolutionalLayer)new Net(inputSize, inputSize)
                    .ConvolutionTranspose(new Size(new[] { outputSize, outputSize }))
                    .Layers[0]);
        }

        public static IEnumerable TestCases
        {
            get
            {
                const int cases = 7;
                for (int i = 1; i < cases; i++)
                for (int o = i; o < cases; o++)
                {
                    yield return new TestCaseData(i, o);
                }
            }
        }
    }
}
