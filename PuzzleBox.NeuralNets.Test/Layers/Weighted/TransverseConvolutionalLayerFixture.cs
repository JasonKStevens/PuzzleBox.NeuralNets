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
            _sut = (ConvolutionalLayer) new Net(inputSize, inputSize)
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

        public static IEnumerable TestCases
        {
            get
            {
                yield return new TestCaseData(1, 1, 1);
                yield return new TestCaseData(1, 2, 2);
            }
        }
    }
}
