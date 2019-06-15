using PuzzleBox.NeuralNets.Algebra;
using System;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public class Relu : ActivationLayerBase
    {
        protected override Func<float, float> _activationFn => ActivationFn.Relu;
        protected override Func<float, float> _gradientFn => ActivationFn.ReluGrad;

        public Relu(Size size) : base(size)
        {
        }
    }

    public static class ReluNetExt
    {
        public static Net Relu(this Net net)
        {
            net.Add(new Relu(net.OutputSize));
            return net;
        }
    }
}
