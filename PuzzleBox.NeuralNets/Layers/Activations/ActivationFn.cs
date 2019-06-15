using System;

namespace PuzzleBox.NeuralNets.Layers.Activations
{
    public static class ActivationFn
    {
        public static float Sigmoid(float x)
        {
            return 1 / ((float)Math.Exp(-x) + 1);
        }

        public static float SigmoidGrad(float x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static float TanH(float x)
        {
            return (float) Math.Tanh(x);
        }

        public static float TanHGrad(float x)
        {
            return 1 - (float) Math.Pow(TanH(x), 2);
        }

        public static float Relu(float x)
        {
            return ReluGrad(x) * x;
        }

        public static float ReluGrad(float x)
        {
            return x > 0 ? 1 : 0;
        }

        public static float LeakyRelu(float x)
        {
            return LeakyReluGrad(x) * x;
        }

        public static float LeakyReluGrad(float x)
        {
            return x > 0 ? 1 : 0.1f;
        }

        public static float ReluSig(float x)
        {
            return ReluSigGrad(x) * x + JaggedSigYIntercept(x);
        }

        private static float JaggedSigYIntercept(float x)
        {
            switch (x)
            {
                case float n when (n < 4):
                    return 0.24f;
                case float n when (n > 4):
                    return 0.76f;
                default:
                    return 0.5f;
            }
        }

        public static float ReluSigGrad(float x)
        {
            return Math.Abs(x) < 4 ? 0.075f : 0.01f;
        }
    }
}