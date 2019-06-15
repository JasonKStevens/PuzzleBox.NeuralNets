using System;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Layers.Weighted
{
    public abstract class WeightsLayerBase : LayerBase, IHaveWeights
    {
        private object _weightsGuard = new object();
        protected Matrix<float> _weights;

        protected WeightsLayerBase(Size inputSize, Size outputSize = null) : base(inputSize, outputSize)
        {
        }

        public Matrix<float> GetWeights()
        {
            return _weights;
        }

        public void SetWeights(Matrix<float> weights)
        {
            lock (_weightsGuard)
                _weights = weights;
        }

        public void UpdateWeights(Func<Matrix<float>, Matrix<float>> updateFunc)
        {
            lock (_weightsGuard)
                _weights = updateFunc(_weights);
        }
    }
}