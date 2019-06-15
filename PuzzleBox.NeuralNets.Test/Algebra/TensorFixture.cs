using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;

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
    }
}
