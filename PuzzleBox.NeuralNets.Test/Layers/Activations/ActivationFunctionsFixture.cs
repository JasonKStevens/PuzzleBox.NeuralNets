using NUnit.Framework;
using PuzzleBox.NeuralNets.Layers.Activations;

namespace PuzzleBox.NeuralNets.Test.Layers.Activations
{
    public class ActivationFunctionsFixture
    {
        [TestCase(0f, 0.5f)]
        [TestCase(1000000, 1)]
        [TestCase(-1000000, 0)]
        public void test_known_sigmoid_values(float z, float value)
        {
            var r = ActivationFn.Sigmoid(z);
            Assert.That(r, Is.EqualTo(value));
        }

        [TestCase(0, 0.25f)]
        [TestCase(1000000, 0)]
        [TestCase(-1000000, 0)]
        public void test_known_sigmoid_gradient_values(float z, float grad)
        {
            var r = ActivationFn.SigmoidGrad(z);
            Assert.That(r, Is.EqualTo(grad));
        }

        [TestCase(-10, 0)]
        [TestCase(0, 0)]
        [TestCase(10, 10)]
        public void test_known_relu_values(float z, float value)
        {
            var r = ActivationFn.Relu(z);
            Assert.That(r, Is.EqualTo(value));
        }

        [TestCase(-10, 0)]
        [TestCase(-0.00001f, 0)]
        [TestCase(0.00001f, 1)]
        [TestCase(10, 1)]
        public void test_known_reul_gradient_values(float z, float grad)
        {
            var r = ActivationFn.ReluGrad(z);
            Assert.That(r, Is.EqualTo(grad));
        }
    }
}
