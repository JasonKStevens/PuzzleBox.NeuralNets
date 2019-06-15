using System;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Test.TestHelpers;

namespace PuzzleBox.NeuralNets.Test
{
    public class FluentFixture
    {
        private Tuple<float[], float>[] _xnorData;
        private Net _sut;

        [SetUp]
        public void Setup()
        {
            _xnorData = new Tuple<float[], float>[]
            {
                new Tuple<float[], float>(new float[] { 0, 0 }, 1),
                new Tuple<float[], float>(new float[] { 0, 1 }, 0),
                new Tuple<float[], float>(new float[] { 1, 0 }, 0),
                new Tuple<float[], float>(new float[] { 1, 1 }, 1)
            };

            _sut = new Net(2)
                .Dense(2)
                .Sigmoid()
                .Dense(1)
                .Sigmoid()
            ;
        }

        [Test]
        public void should_produce_XNOR()
        {
            // Arrange
            var AND_NOR = TestWeights.AND.AppendRows(TestWeights.NOR);
            _sut.SetWeights(new[] { AND_NOR, TestWeights.OR });
            
            // Act & Assert
            foreach (var tuple in _xnorData)
            {
                var activation = _sut.FeedForwards(tuple.Item1);
                var actualOutput = (int)Math.Round(activation, 0);
                Assert.That(actualOutput, Is.EqualTo(tuple.Item2));
            }
        }
    }
}
