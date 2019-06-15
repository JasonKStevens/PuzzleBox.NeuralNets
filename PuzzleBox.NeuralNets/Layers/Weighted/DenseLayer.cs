using System;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Layers.Weighted
{
    public class DenseLayer : WeightsLayerBase
    {
        public DenseLayer(Size inputSize, Size outputSize) : base(inputSize, outputSize)
        {
            _weights = Matrix<float>.Build.Dense(outputSize.TotalLength, inputSize.TotalLength)
                .InsertColumn(0, Vector<float>.Build.Dense(outputSize.TotalLength, 1));  // Bias;
            _weights.InitRandom();
        }

        public DenseLayer(int inputLength, int outputLength)
            : this(new Size(inputLength), new Size(outputLength))
        {
        }

        protected override Tensor FeedForwardsInternal(Tensor input)
        {
            var inputWithBias = Vector<float>.Build.Dense(input.Value.Count + 1, 1);
            inputWithBias.SetSubVector(1, input.Value.Count, input);

            return new Tensor(OutputSize.Clone(), _weights * inputWithBias);
        }

        public override void BackPropagate(TrainingRun trainingRun)
        {
            var inputError = _weights.Transpose() * trainingRun.OutputError.Value;
            trainingRun.InputError = inputError.SubVector(1, inputError.Count - 1);
            trainingRun.WeightsDelta = CalcWeightsDelta(trainingRun.Input, trainingRun.OutputError);
        }

        private Matrix<float> CalcWeightsDelta(Tensor input, Tensor outputError)
        {
            var inputWithBias = Vector<float>.Build.Dense(input.Value.Count + 1, 1);
            inputWithBias.SetSubVector(1, input.Value.Count, input);
            return outputError.Value.OuterProduct(inputWithBias);
        }
    }

    public static class FullyConnNetExt
    {
        public static Net Dense(this Net net, Size outputSize)
        {
            net.Add(new DenseLayer(net.OutputSize, outputSize));
            return net;
        }

        public static Net Dense(this Net net, int outputsize)
        {
            return net.Dense(new Size(outputsize));
        }
    }
}