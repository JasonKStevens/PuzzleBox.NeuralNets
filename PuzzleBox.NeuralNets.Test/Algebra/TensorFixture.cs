using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using System;
using System.Linq;

namespace PuzzleBox.NeuralNets.Test.Algebra
{
    public class TensorFixture
    {
        private Tensor _sut;

        [SetUp]
        public void Setup()
        {
            _sut = new float[] { 0f, 1f };
        }

        [Test]
        public void should_equal_other_tensor_with_same_values()
        {
            Tensor expected = new Tensor(new float[] { 0f, 1f });
            Assert.That(_sut, Is.EqualTo(expected));
        }

        [Test]
        public void should_not_equal_other_tensor_with_different_values()
        {
            Tensor different = new Tensor(new float[] { 0.00001f, 1f });
            Assert.That(_sut, Is.Not.EqualTo(different));
        }

        [Test]
        public void should_have_correct_dimensions_when_created_with_matrix()
        {
            // Arrange
            var original = new float[,] {
                { 1, 2, 3, 4 },
                { 5, 6, 7, 8 },
                { 9, 10 , 11, 12 }
            }.ToMatrix();

            // Act
            var tensor = new Tensor(original);

            // Assert
            Assert.That(tensor.Size.Dimensions.Length, Is.EqualTo(2));
            Assert.That(tensor.Size.Dimensions[0], Is.EqualTo(4));
            Assert.That(tensor.Size.Dimensions[1], Is.EqualTo(3));
        }

        [Test]
        public void should_convert_to_correct_matrix()
        {
            // Arrange
            var original = new float[,] {
                { 1, 2, 3, 4 },
                { 5, 6, 7, 8 },
                { 9, 10 , 11, 12 }
            }.ToMatrix();

            var tensor = new Tensor(original);

            // Act
            var reproduction = tensor.ToMatrix();

            // Assert
            Assert.That(reproduction, Is.EqualTo(original));
        }

        [TestCase(1)]
        [TestCase(2)]
        [TestCase(3)]
        [TestCase(4)]
        [TestCase(6)]
        public void should_support_multiple_dimensions(int dimensionCount)
        {
            // Arrange
            var dimensionSize = 12 / dimensionCount;
            var values = new float[(int)Math.Pow(dimensionSize, dimensionCount)];
            var dimensionSizes = Enumerable.Range(0, dimensionCount)
                .Select(c => dimensionSize)
                .ToArray();

            // Act
            var tensor = new Tensor(new Size(dimensionSizes), values);

            // Assert
            Assert.That(tensor.Size.Dimensions.Length, Is.EqualTo(dimensionCount));
            for (int i = 0; i < dimensionCount; i++)
            {
                Assert.That(tensor.Size.Dimensions[i], Is.EqualTo(12 / dimensionCount));
            }
        }

        [TestCase(2)]
        [TestCase(3)]
        [TestCase(4)]
        [TestCase(6)]
        public void should_split_tensor_by_last_dimension(int dimensionCount)
        {
            // Arrange
            var dimensionSize = 12 / dimensionCount;
            var values = new float[(int)Math.Pow(dimensionSize, dimensionCount)];
            var dimensionSizes = Enumerable.Range(0, dimensionCount)
                .Select(c => dimensionSize)
                .ToArray();
            var tensor = new Tensor(new Size(dimensionSizes), values);

            // Act
            var tensors = tensor.SplitTensorsByLastDimension();

            // Assert
            Assert.That(tensors.Length, Is.EqualTo(dimensionSizes.Last()),
                "The number of tensor arrays should be the same as the size of the last dimension.");

            var reducedDimensions = dimensionSizes.Take(dimensionSizes.Length - 1).ToArray();
            var expectedSize = new Size(reducedDimensions);

            for (int i = 0; i < tensors.Length; i++)
            {
                Assert.That(tensors[i].Size, Is.EqualTo(expectedSize),
                    $"Tensor {i} is not the expected size.");
            }
        }

        public void should_throw_splitting_tensor_with_just_one_dimension()
        {
            // Arrange
            var tensor = new Tensor(new Size(10), new float[10]);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.SplitTensorsByLastDimension());
        }
    }
}
