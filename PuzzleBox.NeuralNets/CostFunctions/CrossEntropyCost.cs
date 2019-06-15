using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.CostFunctions
{
    public class CrossEntropyCost : ICostFunction
    {
        public Vector<float> CalcCost(Vector<float> h, Vector<float> y)
        {
            Guard(h, y);
            return -y.PointwiseMultiply(h.PointwiseLog()) - (1 - y).PointwiseMultiply((1 - h).PointwiseLog());
        }

        public Vector<float> CalcCostGradient(Vector<float> h, Vector<float> y)
        {
            Guard(h, y);
            return -y.PointwiseDivide(h) + (1 - y).PointwiseDivide(1 - h);
        }

        private static void Guard(Vector<float> h, Vector<float> y)
        {
            if (h.Any(e => e <= 0 || e >= 1))
                throw new ArgumentException($"{nameof(CrossEntropyCost)} error: hypothesis '{nameof(h)}' must be between 0 and 1 exclusive.");
            if (y.Any(e => e != 0 && e != 1))
                throw new ArgumentException($"{nameof(CrossEntropyCost)} error: desired output '{nameof(y)}' can only have elements that are 0 or 1. Consider using Softmax");
        }
    }

    public static class CrossEntropyCostTrainerExt
    {
        public static Trainer UseCrossEntropyCost(this Trainer trainer)
        {
            trainer.SetCostFunction(new CrossEntropyCost());
            return trainer;
        }
    }
}
