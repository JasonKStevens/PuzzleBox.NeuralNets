using System;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public abstract class ActivationLayerBase : LayerBase
    {
        protected abstract Func<float, float> _activationFn { get; }
        protected abstract Func<float, float> _gradientFn { get; }

        protected ActivationLayerBase(Size size) : base(size, size)
        {
        }

        protected override Tensor FeedForwardsInternal(Tensor input)
        {
            return input.Map(_activationFn);
        }

        public override void BackPropagate(TrainingRun trainingRun)
        {
            var derivative = trainingRun.Input.Value.Map(_gradientFn);

            trainingRun.InputError = new Tensor(
                InputSize.Clone(),
                trainingRun.OutputError.Value.PointwiseMultiply(derivative)
            );
        }
    }
}
