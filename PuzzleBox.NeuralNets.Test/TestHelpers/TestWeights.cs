using MathNet.Numerics.LinearAlgebra;

namespace PuzzleBox.NeuralNets.Test.TestHelpers
{
    public static class TestWeights
    {
        private static readonly MatrixBuilder<float> M = Matrix<float>.Build;
        public static readonly Matrix<float> NOT = M.DenseOfArray(new float[1, 2] { { 10, -20 } });
        public static readonly Matrix<float> AND = M.DenseOfArray(new float[1, 3] { { -30, 20, 20 } });
        public static readonly Matrix<float> OR = M.DenseOfArray(new float[1, 3] { { -10, 20, 20 } });
        public static readonly Matrix<float> NOR = M.DenseOfArray(new float[1, 3] { { 10, -20, -20 } });
    }
}
