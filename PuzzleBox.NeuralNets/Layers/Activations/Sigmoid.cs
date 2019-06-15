using System;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public class Sigmoid : ActivationLayerBase
    {
        protected override Func<float, float> _activationFn => ActivationFn.Sigmoid;
        protected override Func<float, float> _gradientFn => ActivationFn.SigmoidGrad;

        public Sigmoid(Size size) : base(size)
        {
        }
    }

    public static class SigmoidNetExt
    {
        public static Net Sigmoid(this Net net)
        {
            net.Add(new Sigmoid(net.OutputSize));
            return net;
        }
    }
}
