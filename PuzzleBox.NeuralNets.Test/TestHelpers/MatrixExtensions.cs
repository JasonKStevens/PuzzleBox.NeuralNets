using MathNet.Numerics.LinearAlgebra;

namespace PuzzleBox.NeuralNets.Test.TestHelpers
{
    internal static class MatrixExtensions
    {
        public static Matrix<float> AppendRows(this Matrix<float> matrix, Matrix<float> bottom)
        {
            return matrix.Transpose()
                .Append(bottom.Transpose())
                .Transpose();
        }
    }
}
