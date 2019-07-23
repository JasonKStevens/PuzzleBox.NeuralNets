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
        private readonly int _kernelCount;
        private readonly bool _isTranspose;
        private readonly int[] _strideArray;
        private readonly int[] _paddingArray;

        public ConvolutionalLayer(
            Size inputSize,
            int[] weightLengths,
            int kernelCount,
            int[] strideArray,
            int[] paddingArray)
            : base(inputSize)
        {
            //GuardDimensions(inputSize, outputSize);

            _kernelCount = kernelCount;
            var dimensionCount = inputSize.Dimensions.Length;

            int GetOutputSize(int inSize, int weightLength, int padding, int stride)
            {
                return ((inSize - weightLength) / stride) + padding * 2 + 1;
            }

            var outputDimensions = Enumerable.Range(0, dimensionCount)
                .Select(i => GetOutputSize(inputSize.Dimensions[i], weightLengths[i], paddingArray[i], strideArray[i]))
                .ToArray();

            if (kernelCount > 1)
                outputDimensions = outputDimensions.Concat(new[] { kernelCount }).ToArray();

            OutputSize = new Size(outputDimensions);

            _isTranspose = OutputSize.Length / kernelCount > inputSize.Length;

            _strideArray = strideArray;
            _paddingArray = paddingArray;

            // TODO: Make _weights a Tensor so convolutions aren't limited to 1-2 dimensions
            // var weightLengths = new[] { kernelCount }.Concat(weightLength).ToArray();
            // _weights = Array.CreateInstance(typeof(float), weightLengths);

            var weightColumns = weightLengths[0];
            var weightRows = weightLengths.Length == 1 ? 1 : weightLengths[1];

            _weights = Matrix<float>.Build.Dense(weightRows, weightColumns * _kernelCount);
            _weights.InitRandom();
        }

        protected override Tensor FeedForwardsInternal(Tensor input)
        {
            var weightsArray = _weights.SplitByColumn(_kernelCount)
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
            var outputErrors = trainingRun.OutputError
                .ToMatrices()
                .Select(m => _isTranspose ? m.Rotate180() : m)
                .ToArray();

            trainingRun.WeightsDelta = Matrix<float>.Build.Dense(this._weights.RowCount, this._weights.ColumnCount);

            var weightsDelta = CartesianConvolve(
                outputErrors,
                trainingRun.Input.ToMatrices()
            );

            trainingRun.InputError = MatrixwiseConvolveTrans(
                _weights.SplitByColumn(_kernelCount),
                outputErrors
            );

            for (var r = 0; r < this._weights.RowCount / weightsDelta.RowCount; r++)
            for (var c = 0; c < this._weights.ColumnCount / weightsDelta.ColumnCount; c++)
            {
                trainingRun.WeightsDelta.SetSubMatrix(r, c * weightsDelta.ColumnCount, weightsDelta);
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

            var rowPadding = (InputSize.Dimensions[1] - OutputSize.Dimensions[1] + _weights.RowCount - 1) / 2;
            var columnPadding = (InputSize.Dimensions[0] - OutputSize.Dimensions[0] + _weights.ColumnCount / _kernelCount - 1) / 2;
            
            return Enumerable.Range(0, GetInputKernels())
                .Select(_ => Enumerable.Range(0, _kernelCount)
                    .Select(i => fs[i].ConvolveTranspose(hs[i], rowPadding, columnPadding))
                    .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Add(c))
                    .Divide(_kernelCount)
                )
                .Aggregate((Matrix<float>)null, (m, c) => m == null ? c : m.Append(c));
        }

        private int GetInputKernels()
        {
            return new []{ 1, 2 }.Contains(InputSize.Dimensions.Length) ? 1 : InputSize.Dimensions.Last();
        }

        private static void GuardDimensions(Size inputSize, Size outputSize)
        {
            var isTranspose = outputSize.Length > inputSize.Length;
            var minLength = Math.Min(inputSize.Dimensions.Length, outputSize.Dimensions.Length);

            for (int i = 0; i < minLength; i++)
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

            net.Add(new ConvolutionalLayer(
                inputSize,
                weightLengths,
                kernelCount,
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

            var strideArray = Enumerable.Range(0, inputSize.Dimensions.Length)
                .Select(_ => 1)
                .ToArray();

            var paddingArray = weightLengths
                .Select(l => l - 1)
                .ToArray();

            net.Add(new ConvolutionalLayer(
                inputSize,
                weightLengths,
                kernelCount,
                strideArray,
                paddingArray));
            return net;
        }
    }
}