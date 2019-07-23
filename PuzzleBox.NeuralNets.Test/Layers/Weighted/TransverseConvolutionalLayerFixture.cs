using System;
using System.Collections;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public class TransverseConvolutionalLayerFixture
    {
        private ConvolutionalLayer _sut;

        [TestCaseSource("TestCases")]
        public void should_output_tensor_of_correct_size(int inputSize, int weightLength, int expectedOutputSize)
        {
            // Arrange
            _sut = (ConvolutionalLayer)new Net(inputSize, inputSize)
                .ConvolutionTranspose(new[] { weightLength, weightLength })
                .Layers[0];

            var input = new float[inputSize, inputSize].ToMatrix();

            // Act
            var output = _sut.FeedForwards(new Tensor(input));

            // Assert
            Assert.That(output.Size.Dimensions.Length, Is.EqualTo(2));
            Assert.That(output.Size.Dimensions[0], Is.EqualTo(expectedOutputSize));
            Assert.That(output.Size.Dimensions[1], Is.EqualTo(expectedOutputSize));
        }

        [TestCaseSource("TestCases")]
        public void should_backpropagate_tensor_of_correct_size(int inputSize, int weightLength, int outputSize)
        {
            // Arrange
            _sut = (ConvolutionalLayer)new Net(inputSize, inputSize)
                .ConvolutionTranspose(new[] { weightLength, weightLength })
                .Layers[0];

            var trainingRun = new TrainingRun(1)
            {
                Input = new float[inputSize, inputSize].ToMatrix(),
                Output = _sut.FeedForwards(new float[inputSize, inputSize].ToMatrix()),
                OutputError = new float[outputSize, outputSize].ToMatrix()
            };

            // Act
            _sut.BackPropagate(trainingRun);

            // Assert
            Assert.That(trainingRun.InputError.Size, Is.EqualTo(trainingRun.Input.Size));
        }

        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        public void should_backpropagate_tensor_of_correct_kernel_size(int kernelSize)
        {
            // Arrange
            _sut = (ConvolutionalLayer)new Net(1, 1)
                .ConvolutionTranspose(new[] { 5, 5 }, kernelSize)
                .Layers[0];
            var input = new float[,] { { 0 } }.ToMatrix();

            var trainingRun = new TrainingRun(1)
            {
                Input = input,
                Output = _sut.FeedForwards(input),
                OutputError = new Tensor(new Size(5, 5, kernelSize), new float[5 * 5 * kernelSize])
            };

            // Act
            _sut.BackPropagate(trainingRun);

            // Assert
            Assert.That(trainingRun.InputError.Size, Is.EqualTo(trainingRun.Input.Size));
        }

        public static IEnumerable TestCases
        {
            get
            {
                yield return new TestCaseData(1, 1, 1);
                yield return new TestCaseData(1, 2, 2);
                yield return new TestCaseData(1, 3, 3);
                yield return new TestCaseData(2, 2, 3);
                yield return new TestCaseData(2, 3, 4);
                yield return new TestCaseData(2, 4, 5);
                yield return new TestCaseData(3, 3, 5);
                yield return new TestCaseData(3, 4, 6);
                yield return new TestCaseData(3, 5, 7);
                yield return new TestCaseData(3, 6, 8);
            }
        }
    }
}
