using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Test.TestHelpers
{
    public static class TrainingData
    {
        public static (Tensor input, Tensor output)[] NOT = new (Tensor input, Tensor output)[]
        {
                (0, 1),
                (1, 0)
        };

        public static (Tensor input, Tensor output)[] AND = new (Tensor input, Tensor output)[]
        {
            (new float[] { 0, 0 }, 0),
            (new float[] { 0, 1 }, 0),
            (new float[] { 1, 0 }, 0),
            (new float[] { 1, 1 }, 1)
        };
        
        public static (Tensor input, Tensor output)[] OR = new (Tensor input, Tensor output)[]
        {
            (new float[] { 0, 0 }, 0),
            (new float[] { 0, 1 }, 1),
            (new float[] { 1, 0 }, 1),
            (new float[] { 1, 1 }, 1)
        };

        public static (Tensor input, Tensor output)[] XNOR = new (Tensor input, Tensor output)[]
        {
            (new float[] { 0, 0 }, 1),
            (new float[] { 0, 1 }, 0),
            (new float[] { 1, 0 }, 0),
            (new float[] { 1, 1 }, 1)
        };

        public static (Tensor input, Tensor output)[] Lines2d = new (Tensor input, Tensor output)[]
        {
            (
                new float[,]
                {
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 1, 0, 0, 0, 0 },
                    { 1, 0, 0, 0, 0 },
                    { 1, 0, 0, 0, 0 },
                    { 1, 0, 0, 0, 0 },
                    { 1, 0, 0, 0, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0, 1 },
                },
                1
            )
        };

        public static (Tensor input, Tensor output)[] Lines2dPadded = new (Tensor input, Tensor output)[]
        {
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0 },
                    { 1, 1, 1, 1, 1 },
                    { 0, 0, 0, 0, 0 },
                },
                0
            ),
            (
                new float[,]
                {
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0 },
                },
                1
            ),
            (
                new float[,]
                {
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0 },
                },
                1
            ),
        };
    }
}
