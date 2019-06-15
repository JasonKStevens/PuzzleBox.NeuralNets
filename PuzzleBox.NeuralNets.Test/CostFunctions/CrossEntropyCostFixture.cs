using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;

namespace PuzzleBox.NeuralNets.Test.CostFunctions
{
    public class CrossEntropyCostFixture
    {
        private CrossEntropyCost _sut;
        private const float FloatTolerance = 0.0001f;

        [SetUp]
        public void SetUp()
        {
            _sut = new CrossEntropyCost();
        }

        [TestCase(0.99999f, 1, 0)]
        [TestCase(0.00001f, 0, 0)]
        [TestCase(0.5f, 1, 0.6931f)]
        public void test_known_cost_values(float h, float y, float expected)
        {
            var c = _sut.CalcCost(h.ToVector(), y.ToVector());
            Assert.That(c[0], Is.EqualTo(expected).Within(FloatTolerance));
        }

        [TestCase(0.9f, 0, 1/0.1f)]
        [TestCase(0.5f, 0, 1/0.5f)]
        [TestCase(0.1f, 0, 1/0.9f)]
        [TestCase(0.9f, 1, -1/0.9f)]
        [TestCase(0.5f, 1, -1/0.5f)]
        [TestCase(0.1f, 1, -1/0.1f)]
        public void test_known_cost_gradient_values(float h, float y, float expected)
        {
            var c = _sut.CalcCostGradient(h.ToVector(), y.ToVector());
            Assert.That(c[0], Is.EqualTo(expected).Within(FloatTolerance));
        }

        [TestCase(0.9f, 0)]
        [TestCase(0.5f, 0)]
        [TestCase(0.1f, 0)]
        [TestCase(0.9f, 1)]
        [TestCase(0.5f, 1)]
        [TestCase(0.1f, 1)]
        public void test_cost_gradient_is_gradient_of_cost(float h, float y)
        {
            // Arrange
            const float dh = 0.002f;
            var dc = _sut.CalcCost((h + dh/2).ToVector(), y.ToVector())
                   - _sut.CalcCost((h - dh/2).ToVector(), y.ToVector());
            var g = dc/dh;
            
            // Act
            var c = _sut.CalcCostGradient(h.ToVector(), y.ToVector());

            // Assert
            Assert.That(c[0], Is.EqualTo(g[0]).Within(10 * FloatTolerance));  // More tolerance for numerical errors
        }

        [TestCase(0.9f, 1, false)]
        [TestCase(1, 1, true)]
        [TestCase(2, 1, true)]
        [TestCase(0.1f, 1, false)]
        [TestCase(0, 1, true)]
        [TestCase(-1, 1, true)]
        [TestCase(0.5f, 0, false)]
        [TestCase(0.5f, 1, false)]
        [TestCase(0.5f, 0.5f, true)]
        public void should_throw_when_values_outside_of_allowed_range(float h, float y, bool shouldThrow)
        {
            if (shouldThrow)
            {
                Assert.That(() => _sut.CalcCost(h.ToVector(), y.ToVector()), Throws.Exception);
                Assert.That(() => _sut.CalcCostGradient(h.ToVector(), y.ToVector()), Throws.Exception);
            }
            else
            {
                Assert.That(() => _sut.CalcCost(h.ToVector(), y.ToVector()), Throws.Nothing);
                Assert.That(() => _sut.CalcCostGradient(h.ToVector(), y.ToVector()), Throws.Nothing);
            }
        }
    }
}
