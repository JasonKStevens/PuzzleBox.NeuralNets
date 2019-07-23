using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace PuzzleBox.NeuralNets.Algebra
{
    public class Tensor : IEquatable<Tensor>
    {
        public Size Size { get; }
        public Vector<float> Value { get; }

        public Tensor(Size size, Vector<float> value)
        {
            if (size.Length != value.Count)
                throw new ArgumentOutOfRangeException("Total tensor length must match vector length.");

            Size = size;
            Value = value;
        }

        public Tensor(params float[] values) : this(Vector<float>.Build.Dense(values))
        {
        }

        public Tensor(Matrix<float> matrix) :
            this(
                new Size(matrix.ColumnCount, matrix.RowCount),
                Vector<float>.Build.DenseOfArray(matrix.ToColumnMajorArray())
            )
        {
        }

        public Tensor(Vector<float> value) : this(new Size(value.Count), value)
        {
        }

        public Tensor(Size size, float[] value) : this(size, Vector<float>.Build.DenseOfArray(value))
        {
        }

        public Tensor Map(Func<float, float> func)
        {
            return new Tensor(Size.Clone(), Value.Map(func));
        }

        /// <summary>
        /// Returns the tensor as a matrix, iff the tensor is 2d.
        /// </summary>
        public Matrix<float> ToMatrix()
        {
            var dims = Size.Dimensions;
            if (dims.Length < 1)
                throw new InvalidCastException($"Cannot convert a {dims.Length}d tensor into a 2d matrix.");

            var singleLayer3D = (dims.Length == 3 && dims[2] == 1);
            if (dims.Length > 2 && !singleLayer3D)
                throw new InvalidCastException($"Cannot convert a {dims.Length}d tensor into a 2d matrix.");

            var columns = dims[0];
            var rows = dims.Length > 1 ? dims[1] : 1;
            var matrix = Matrix<float>.Build.Dense(rows, columns);

            for (int c = 0; c < columns; c++)
            {
                matrix.SetColumn(c, Value.SubVector(c * rows, rows));
            }

            return matrix;
        }

        public Matrix<float>[] ToMatrices()
        {
            var kernelCount = Size.Dimensions.Length == 3 ? Size.Dimensions[2] : 1;

            if (!new [] { 1, 2, 3 }.Contains(Size.Dimensions.Length))
                throw new ArgumentException($"Tensor's size must be 1d, 2d or 3d (with kernels) to convert to matrices.");

            var cols = Size.Dimensions[0];
            var rows = Size.Dimensions.Length > 1 ? Size.Dimensions[1] : 1;

            var subVectorLength = cols * rows;

            if (Value.Count % subVectorLength != 0)
                throw new ArgumentException($"Vector does not evenly split into desired matrix size.");

            var matrices = Enumerable.Range(0, kernelCount)
                .Select(f => Value.SubVector(f * subVectorLength, subVectorLength))
                .Select(ev => Matrix<float>.Build.DenseOfColumnMajor(rows, cols, ev))
                .ToArray();
            return matrices;
        }

        public override string ToString()
        {
            return Value.ToString();
        }

        public bool Equals(Tensor other)
        {
            if (other == null)
                return false;

            return
                this.Size.Equals(other.Size) &&
                this.Value.Equals(other.Value);
        }

        /// <summary>
        /// Splits tensor into multiple by peeling off last dimension.
        /// This is used to get multiple kernels from a single tensor for example.
        /// </summary>
        /// <returns>Tensor array reduce by one dimension.</returns>
        public Tensor[] SplitTensorsByLastDimension()
        {
            if (Size.Dimensions.Length == 1)
                throw new InvalidOperationException("Cannot split tensor by last dimension with just one dimension.");

            var newDimensions = Size.Dimensions.Take(Size.Dimensions.Length - 1).ToArray();
            var newSize = new Size(newDimensions);
            var lastDimension = Size.Dimensions.Last();

            var tensors = Enumerable.Range(0, lastDimension)
                .Select(i => Value.SubVector(i * newSize.Length, newSize.Length))
                .Select(v => new Tensor(newSize, v))
                .ToArray();
            return tensors;
        }

        public Matrix<float>[] SplitIntoMatricesByLastDimension()
        {
            return SplitTensorsByLastDimension()
                .Select(t => t.ToMatrix())
                .ToArray();
        }

        public static implicit operator Matrix<float>(Tensor tensor)
        {
            return tensor.ToMatrix();
        }

        public static implicit operator Tensor(Matrix<float> matrix)
        {
            return new Tensor(matrix);
        }

        public static implicit operator float[,](Tensor tensor)
        {
            return tensor.ToMatrix().AsArray();
        }

        public static implicit operator Tensor(float[,] matrixArray)
        {
            return new Tensor(matrixArray.ToMatrix());
        }

        public static implicit operator Vector<float>(Tensor tensor)
        {
            return tensor.Value;
        }

        public static implicit operator Tensor(Vector<float> vector)
        {
            return new Tensor(vector);
        }

        public static implicit operator float[] (Tensor tensor)
        {
            return tensor.Value.AsArray();
        }

        public static implicit operator Tensor(float[] floatArray)
        {
            return new Tensor(floatArray);
        }

        public static implicit operator float (Tensor tensor)
        {
            return tensor.Value[0];
        }

        public static implicit operator Tensor(float @float)
        {
            return new Tensor(@float);
        }
    }
}
