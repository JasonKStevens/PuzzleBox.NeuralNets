using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Layers.Weighted
{
    /// <summary>
    /// 2d (or 1d) convolutional layer with no padding, to reduce output size.
    /// </summary>
    public class ConvolutionalLayer : WeightsLayerBase
    {
        private readonly bool _isTranspose;
        private readonly int[] _strideArray;
        private readonly int[] _paddingArray;

        public ConvolutionalLayer(
            Size inputSize,
            Size outputSize,
            int[] weightLength,
            int[] strideArray,
            int[] paddingArray)
            : base(inputSize, outputSize)
        {
            GuardDimensions(outputSize, inputSize);

            _isTranspose = outputSize.Length > inputSize.Length;

            _strideArray = strideArray;
            _paddingArray = paddingArray;

            // TODO: Fix current limitation of just 1-2 convolution dimensions due to base-class _weights.
            // var weightLengths = new[] { kernelCount }.Concat(weightLength).ToArray();
            // _weights = Array.CreateInstance(typeof(float), weightLengths);

            var weightColumns = weightLength[0];
            var weightRows = weightLength.Length == 1 ? 1 : weightLength[1];

            _weights = Matrix<float>.Build.Dense(weightRows, weightColumns * outputSize.KernelCount);
            _weights.InitRandom();
        }

        protected override Tensor FeedForwardsInternal(Tensor input)
        {
            var weightsArray = _weights.SplitByColumn(OutputSize.KernelCount)
                .Select(w => _isTranspose ? w.Rotate180() : w)
                .ToArray();

            var rows = CartesianConvolve(
                weightsArray,
                input.ToMatrices()
            )
            .ToColumnMajorArray();

            return new Tensor(OutputSize.Clone(), rows);
        }

        public override void BackPropagate(TrainingRun trainingRun)
        {
            var outputError = trainingRun.OutputError
                .ToMatrices()
                .Select(m => _isTranspose ? m.Rotate180() : m)
                .ToArray();

            var weightsDelta = CartesianConvolve(
                outputError,
                trainingRun.Input.ToMatrices()
            );

            trainingRun.InputError = MatrixwiseConvolveTrans(
                _weights.SplitByColumn(OutputSize.KernelCount),
                outputError
            );

            // Ensure weightsDelta is the same size as _weights
            // Currently done only for enlarging, TODO: Anything more needed here?
            trainingRun.WeightsDelta = Matrix<float>.Build.Dense(this._weights.RowCount, this._weights.ColumnCount);

            for (var r = 0; r < this._weights.RowCount / weightsDelta.RowCount; r++)
            for (var c = 0; c < this._weights.ColumnCount / weightsDelta.ColumnCount; c++)
            {
                trainingRun.WeightsDelta.SetSubMatrix(r, c, weightsDelta);
            }
        }

        private Matrix<float> CartesianConvolve(Matrix<float>[] fs, Matrix<float>[] gs)
        {
            var strideRows = _strideArray.Length > 1 ? _strideArray[1] : 1;
            var paddingRows = _paddingArray.Length > 1 ? _paddingArray[1] : 1;

            return fs
                .Select(f => gs
                    .Select(g => f.Convolve(g, strideRows, _strideArray[0], paddingRows, _paddingArray[0]))
                    .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Add(c))
                    .Divide(gs.Length)
                )
                .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Append(c));
        }

        private Matrix<float> MatrixwiseConvolveTrans(Matrix<float>[] fs, Matrix<float>[] hs)
        {
            if (fs.Length != hs.Length)
                throw new ArgumentException("Feature count doesn't match.");

            // TODO: Find formula for padding in transpose convolution
            var rowPadding = (InputSize.Dimensions[1] - OutputSize.Dimensions[1] + _weights.RowCount - 1) / 2;
            var columnPadding = (InputSize.Dimensions[0] - OutputSize.Dimensions[0] + _weights.ColumnCount - 1) / 2;

            return Enumerable.Range(0, InputSize.KernelCount)
                .Select(_ => Enumerable.Range(0, OutputSize.KernelCount)
                    .Select(i => fs[i].ConvolveTranspose(hs[i], rowPadding, columnPadding))
                    .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Add(c))
                    .Divide(OutputSize.KernelCount)
                )
                .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Append(c));
        }

        private static void GuardDimensions(Size outputSize, Size inputSize)
        {
            if (inputSize.Dimensions.Length != outputSize.Dimensions.Length)
                throw new ArgumentException("The number of input and output dimensions must match.");

            var isTranspose = outputSize.Length > inputSize.Length;

            for (int i = 0; i < inputSize.Dimensions.Length; i++)
            {
                var inputDimension = inputSize.Dimensions[i];
                var outputDimension = outputSize.Dimensions[i];

                if (inputDimension < 1)
                    throw new ArgumentException($"Input dimension {i} cannot be less that 1.");

                if (outputDimension < 1)
                    throw new ArgumentException($"Output dimension {i} cannot be less that 1.");

                if (isTranspose && outputDimension < inputDimension)
                    throw new ArgumentException($"Output dimension {i} cannot be less that input dimension for transpose convolutions.");

                if (!isTranspose && inputDimension < outputDimension)
                    throw new ArgumentException($"Input dimension {i} cannot be less that output dimension for transpose convolutions.");
            }
        }
    }

    public static class ConvolutionalLayerNetExt
    {
        public static Net Convolution(
            this Net net,
            int[] weightLengths,
            int kernelCount = 1,
            int[] strideArray = null)
        {
            var inputSize = net.OutputSize;
            var dimensionCount = inputSize.Dimensions.Length;

            strideArray = strideArray ?? Enumerable.Range(0, dimensionCount)
                .Select(_ => 1)
                .ToArray();

            if (strideArray.Length != dimensionCount)
                throw new ArgumentOutOfRangeException($"Stride array dimensions ({strideArray.Length}) must be the same as the input ({dimensionCount}).");

            var paddingArray = weightLengths
                .Select(l => (l - 1) / 2)
                .ToArray();

            int GetOutputSize(int inSize, int weightLength, int padding, int stride)
            {
                return ((inSize - weightLength) / stride) + padding * 2 + 1;
            }

            var outputDimensions = Enumerable.Range(0, dimensionCount)
                .Select(i => GetOutputSize(inputSize.Dimensions[i], weightLengths[i], paddingArray[i], strideArray[i]))
                .ToArray();

            var outputSize = new Size(outputDimensions, kernelCount);

            net.Add(new ConvolutionalLayer(
                inputSize,
                outputSize,
                weightLengths,
                strideArray,
                paddingArray));
            return net;
        }

        public static Net ConvolutionTranspose(
            this Net net,
            int[] weightLengths,
            int kernelCount = 1)
        {
            var inputSize = net.OutputSize;
            var dimensionCount = inputSize.Dimensions.Length;

            var strideArray = Enumerable.Range(0, dimensionCount)
                .Select(_ => 1)
                .ToArray();

            var paddingArray = weightLengths
                .Select(l => l - 1)
                .ToArray();

            int GetOutputSize(int inSize, int weightLength, int padding, int stride)
            {
                return ((inSize - weightLength) / stride) + padding * 2 + 1;
            }

            var outputDimensions = Enumerable.Range(0, dimensionCount)
                .Select(i => GetOutputSize(inputSize.Dimensions[i], weightLengths[i], paddingArray[i], strideArray[i]))
                .ToArray();

            var outputSize = new Size(outputDimensions, kernelCount);

            net.Add(new ConvolutionalLayer(
                inputSize,
                outputSize,
                weightLengths,
                strideArray,
                paddingArray));
            return net;
        }
    }
}