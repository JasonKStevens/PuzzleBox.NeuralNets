using PuzzleBox.NeuralNets.Algebra;
using System;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public class ReluSig : ActivationLayerBase
    {
        protected override Func<float, float> _activationFn => ActivationFn.ReluSig;
        protected override Func<float, float> _gradientFn => ActivationFn.ReluSigGrad;

        public ReluSig(Size size) : base(size)
        {
        }
    }

    public static class ReluSigNetExt
    {
        public static Net ReluSig(this Net net)
        {
            net.Add(new ReluSig(net.OutputSize));
            return net;
        }
    }
}
