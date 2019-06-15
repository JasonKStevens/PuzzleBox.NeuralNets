using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Layers;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets
{
    public class Net : LayerBase
    {
        public List<ILayer> Layers { get; } = new List<ILayer>();

        public Net(Size inputSize) : base(inputSize, inputSize)
        {
            InputSize = inputSize;
        }

        public Net(params int[] inputSize) : this(new Size(inputSize))
        {
        }

        public Net Add(ILayer layer)
        {
            // Adjust output size
            OutputSize = layer.OutputSize;

            Layers.Add(layer);
            return this;
        }

        public Matrix<float>[] GetWeights()
        {
            return Layers
                .OfType<IHaveWeights>()
                .Select(l => l.GetWeights())
                .ToArray();
        }

        public void SetWeights(Matrix<float>[] weights)
        {
            var weightedLayers = Layers.OfType<IHaveWeights>().ToArray();

            for (int i = 0; i < weightedLayers.Length; i++)
            {
                weightedLayers[i].SetWeights(weights[i]);
            }
        }

        protected override Tensor FeedForwardsInternal(Tensor input)
        {
            return Layers.Aggregate(input, (m, l) => l.FeedForwards(m));
        }

        public TrainingRun FeedForwardsTraining(Tensor input)
        {
            var trainingRun = new TrainingRun(Layers.Count)
            {
                Input = input,
                BatchSize = 1
            };

            // TODO: Guard clauses for when Tensors are unexpected shape
            for (int i = 0; i < Layers.Count; i++)
            {
                trainingRun.Counter = i;
                trainingRun.Output = Layers[i].FeedForwards(trainingRun.Input);
            }

            return trainingRun;
        }

        public override void BackPropagate(TrainingRun trainingRun)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                trainingRun.Counter = i;
                Layers[i].BackPropagate(trainingRun);
            }
        }

        public void ApplyTraining(TrainingRun trainingRun, float learningRate)
        {
            Parallel.For(0, Layers.Count, i =>
            {
                if (Layers[i] is IHaveWeights layer)
                {
                    // TODO: Regularization var regularisation = layer.GetWeights() / trainingRun.BatchSize;
                    var delta = trainingRun.WeightsDeltas[i] * (learningRate / trainingRun.BatchSize); // + regularisation;

                    layer.UpdateWeights(weights => weights - delta);
                }
            });
        }
    }
}
