using System;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public class TanH : ActivationLayerBase
    {
        protected override Func<float, float> _activationFn => ActivationFn.TanH;
        protected override Func<float, float> _gradientFn => ActivationFn.TanHGrad;

        public TanH(Size size) : base(size)
        {
        }
    }

    public static class TanHNetExt
    {
        public static Net TanH(this Net net)
        {
            net.Add(new TanH(net.OutputSize));
            return net;
        }
    }
}
