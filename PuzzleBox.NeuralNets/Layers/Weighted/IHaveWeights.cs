using System;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Layers.Weighted
{
    public interface IHaveWeights
    {
        Matrix<float> GetWeights();
        void SetWeights(Matrix<float> weights);
        void UpdateWeights(Func<Matrix<float>, Matrix<float>> updateFunc);
    }

    public static class WeightsExtensions
    {
        public static void SetWeights(this IHaveWeights layer, float[,] weights)
        {
            layer.SetWeights(weights.ToMatrix());
        }
    }
}
