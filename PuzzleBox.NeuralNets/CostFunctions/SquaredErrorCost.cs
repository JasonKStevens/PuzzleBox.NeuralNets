using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.CostFunctions
{
    public class QuadraticCost : ICostFunction
    {
        public Vector<float> CalcCost(Vector<float> h, Vector<float> y)
        {
            return (h - y).PointwiseMultiply(h - y) / 2;
        }

        public Vector<float> CalcCostGradient(Vector<float> h, Vector<float> y)
        {
            return h - y;
        }
    }

    public static class QuadraticCostTrainerExt
    {
        public static Trainer UseQuadraticCost(this Trainer trainer)
        {
            trainer.SetCostFunction(new QuadraticCost());
            return trainer;
        }
    }
}
