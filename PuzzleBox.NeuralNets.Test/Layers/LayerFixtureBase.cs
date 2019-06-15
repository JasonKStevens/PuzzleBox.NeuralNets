using System.Linq;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers;

namespace PuzzleBox.NeuralNets.Test.Layers
{
    public abstract class LayerFixtureBase<T>
        where T : ILayer
    {
        protected T _sut;

        [Test]
        public void should_emit_forwards()
        {
            // Arrange
            var values = Enumerable.Range(1, _sut.InputSize.Length)
                .Select(x => (float)x)
                .ToArray();
            var input = new Tensor(_sut.InputSize.Clone(), values);

            // Act
            var output = _sut.FeedForwards(input);

            // Assert
            Assert.That(output, Is.Not.Null);
        }
    }
}
