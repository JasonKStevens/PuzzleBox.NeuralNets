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
        private readonly int _rowStride;
        private readonly int _columnStride;
        private readonly int _rowPadding;
        private readonly int _columnPadding;
        private readonly bool _isTranspose;

        public ConvolutionalLayer(
            Size inputSize,
            Size outputSize,
            int weightRows,
            int weightColumns,
            int rowStride = 1,
            int columnStride = 1,
            int rowPadding = 0,
            int columnPadding = 0)
            : base(inputSize, outputSize)
        {
            _rowStride = rowStride;
            _columnStride = columnStride;
            _rowPadding = rowPadding;
            _columnPadding = columnPadding;

            _isTranspose = outputSize.Length > inputSize.Length;

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

        private Matrix<float> CartesianConvolve(Matrix<float>[] fs, Matrix<float>[] gs)
        {
            //return (
            //    from f in fs
            //    from g in gs
            //    select f.Convolve(g, _rowStride, _columnStride, _rowPadding, _columnPadding)
            //).Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Append(c));
            return fs
                .Select(f => gs
                    .Select(g => f.Convolve(g, _rowStride, _columnStride, _rowPadding, _columnPadding))
                    .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Add(c))
                    .Divide(gs.Length)
                )
                .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Append(c));
        }

        public override void BackPropagate(TrainingRun trainingRun)
        {
            var outputError = trainingRun.OutputError
                .ToMatrices()
                .Select(m => _isTranspose ? m.Rotate180() : m)
                .ToArray();

            trainingRun.WeightsDelta = CartesianConvolve(
                outputError,
                trainingRun.Input.ToMatrices()
            );

            trainingRun.InputError = MatrixwiseConvolveTrans(
                _weights.SplitByColumn(OutputSize.KernelCount),
                trainingRun.WeightsDelta.SplitByColumn(OutputSize.KernelCount)
            );
        }

        private Matrix<float> MatrixwiseConvolveTrans(Matrix<float>[] fs, Matrix<float>[] hs)
        {
            if (fs.Length != hs.Length)
                throw new ArgumentException("Feature count doesn't match.");

            // TODO: Find formula for padding in transpose convolution
            var rowPadding = _isTranspose ? 0 : (int?)null;
            var columnPadding = _isTranspose ? 0 : (int?)null;

            return Enumerable.Range(0, OutputSize.KernelCount)
                .Select(i => fs[i].ConvolveTranspose(hs[i], rowPadding, columnPadding))
                .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Add(c));
        }
    }

    public static class ConvolutionalLayerNetExt
    {
        public static Net Convolution(
            this Net net,
            Size outputSize,
            int weightRows,
            int weightColumns
            )
        {
            var inputSize = net.OutputSize;
            GuardDimensions(outputSize, inputSize);

            var columnPadding = (weightColumns - 1) / 2;
            var rowPadding = (weightRows - 1) / 2;

            var is2d = inputSize.Dimensions.Length > 1;

            int columnStride = inputSize.Dimensions[0] / outputSize.Dimensions[0];
            int rowStride = is2d ? inputSize.Dimensions[1] / outputSize.Dimensions[1] : 1;

            net.Add(new ConvolutionalLayer(
                inputSize,
                outputSize,
                weightRows,
                weightColumns,
                rowStride,
                columnStride,
                rowPadding,
                columnPadding));
            return net;
        }

        public static Net ConvolutionTranspose(
            this Net net,
            Size outputSize)
        {
            var inputSize = net.OutputSize;
            GuardDimensions(outputSize, inputSize, isTranspose: true);

            var is2d = inputSize.Dimensions.Length > 1;

            var columnPadding = outputSize.Dimensions[0] - inputSize.Dimensions[0];
            var rowPadding = is2d ? outputSize.Dimensions[1] - inputSize.Dimensions[1] : 1;

            var weightColumns = columnPadding + 1;
            var weightRows = rowPadding + 1;

            net.Add(new ConvolutionalLayer(
                inputSize,
                outputSize,
                weightRows,
                weightColumns,
                rowStride: 1,
                columnStride: 1,
                rowPadding: rowPadding,
                columnPadding: columnPadding));
            return net;
        }

        private static void GuardDimensions(Size outputSize, Size intputSize, bool isTranspose = false)
        {
            if (intputSize.Dimensions.Length != outputSize.Dimensions.Length)
                throw new ArgumentException("The number of input and output dimensions must match.");

            for (int i = 0; i < intputSize.Dimensions.Length; i++)
            {
                var inputDimension = intputSize.Dimensions[i];
                var outputDimension = outputSize.Dimensions[i];

                if (isTranspose && outputDimension < inputDimension)
                    throw new ArgumentException($"Output dimension {i} cannot be less that input dimension for transpose convolutions.");

                if (!isTranspose && inputDimension < outputDimension)
                    throw new ArgumentException($"Input dimension {i} cannot be less that output dimension for transpose convolutions.");
            }
        }
    }
}