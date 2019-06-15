using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Training;
using System;

namespace PuzzleBox.NeuralNets.Layers
{
    public abstract class LayerBase : ILayer
    {
        public Size InputSize { get; protected set; }
        public Size OutputSize { get; protected set; }

        protected LayerBase(Size inputSize, Size outputSize = null)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        public Tensor FeedForwards(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (!input.Size.Equals(InputSize))
                throw new ArgumentException($"Input tensor is of a different shape to the layer's declared input.");

            var output = FeedForwardsInternal(input);

            if (output == null)
                throw new InvalidOperationException($"Layer cannot output null in its feed-forward operation.");
            if (!output.Size.Equals(OutputSize))
                throw new InvalidOperationException($"Output tensor is of a different shape to the layer's declared output.");

            return output;
        }

        protected abstract Tensor FeedForwardsInternal(Tensor input);
        public abstract void BackPropagate(TrainingRun trainingRun);
    }

    public static class LayerBaseExtensions
    {
        public static float[] FeedForwards(this ILayer layer, params float[] input)
        {
            return layer.FeedForwards(input);
        }
    }
}