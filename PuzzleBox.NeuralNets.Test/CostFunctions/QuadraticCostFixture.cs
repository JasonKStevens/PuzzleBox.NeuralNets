using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;

namespace PuzzleBox.NeuralNets.Test.CostFunctions
{
    public class QuadraticCostFixture
    {
        private QuadraticCost _sut;

        [SetUp]
        public void SetUp()
        {
            _sut = new QuadraticCost();
        }

        [TestCase(-10, 0, 50)]
        [TestCase(10, 0, 50)]
        [TestCase(-10, 10, 200)]
        [TestCase(10, 10, 0)]
        public void test_known_cost_values(float h, float y, float expected)
        {
            var c = _sut.CalcCost(h.ToVector(), y.ToVector());
            Assert.That(c[0], Is.EqualTo(expected));
        }

        [TestCase(-10, 0, -10)]
        [TestCase(10, 0, 10)]
        [TestCase(-10, 10, -20)]
        [TestCase(10, 10, 0)]
        public void test_known_cost_gradient_values(float h, float y, float expected)
        {
            var c = _sut.CalcCostGradient(h.ToVector(), y.ToVector());
            Assert.That(c[0], Is.EqualTo(expected));
        }

        [TestCase(-10, -10)]
        [TestCase(0, -10)]
        [TestCase(10, -10)]
        [TestCase(-10, 0)]
        [TestCase(0, 0)]
        [TestCase(10, 0)]
        [TestCase(-10, 10)]
        [TestCase(0, 10)]
        [TestCase(10, 10)]
        public void test_cost_gradient_is_gradient_of_cost(float h, float y)
        {
            // Arrange
            const float dh = 0.002f;
            var dc = _sut.CalcCost((h + dh / 2).ToVector(), y.ToVector())
                   - _sut.CalcCost((h - dh / 2).ToVector(), y.ToVector());
            var g = dc / dh;

            // Act
            var c = _sut.CalcCostGradient(h.ToVector(), y.ToVector());

            // Assert
            Assert.That(c[0], Is.EqualTo(g[0]).Within(0.1f));
        }
    }
}
