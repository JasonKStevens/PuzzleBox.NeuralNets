using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

namespace PuzzleBox.NeuralNets.Algebra
{
    public static class MathNetExtensions
    {
        private static Random Rnd = new Random(DateTime.Now.Millisecond);

        public static void SetRandomSeed(int seed)
        {
            Rnd = new Random(seed);
        }

        public static Vector<float> ToVector(this float[] floatArray)
        {
            return Vector<float>.Build.Dense(floatArray);
        }

        public static Vector<float> ToVector(this float floatArray)
        {
            return Vector<float>.Build.Dense(new float[] { floatArray });
        }

        public static Matrix<float> ToMatrix(this float[,] floatArray)
        {
            return Matrix<float>.Build.DenseOfArray(floatArray);
        }

        public static Matrix<float>[] SplitByColumn(this Matrix<float> matrix, int numColumns)
        {
            if (matrix.ColumnCount % numColumns != 0)
                throw new ArgumentException($"Matrix does not evenly split by column into desired size.");

            var width = matrix.ColumnCount / numColumns;

            return Enumerable.Range(0, numColumns)
                .Select(f => matrix.SubMatrix(0, matrix.RowCount, f * width, width))
                .ToArray();
        }

        public static void InitRandom(this Matrix<float> matrix)
        {
            var ε = (float)(Math.Sqrt(6) / Math.Sqrt(matrix.ColumnCount + matrix.RowCount));

            for (var r = 0; r < matrix.RowCount; r++)
            for (var c = 0; c < matrix.ColumnCount; c++)
                matrix[r, c] = ε * ((float)Rnd.NextDouble() * 2 - 1);
        }

        /// <summary>
        /// Transpose convolve two matrices f*g.
        /// <see cref="Convolve"/>.
        /// </summary>
        public static Matrix<float> ConvolveTranspose(
            this Matrix<float> f,
            Matrix<float> g,
            int? rowPadding = null,
            int? columnPadding = null)
        {
            return f
                .Rotate180()
                .Convolve(
                    g,
                    rowPadding: rowPadding ?? f.RowCount - 1,
                    columnPadding: columnPadding ?? f.ColumnCount - 1
                );
        }

        /// <summary>
        /// Convolve two matrices f*g.
        /// <see cref="ConvolveTranspose"/>.
        /// </summary>
        public static Matrix<float> Convolve(
            this Matrix<float> f,
            Matrix<float> g,
            int rowStride = 1,
            int columnStride = 1,
            int rowPadding = 0,
            int columnPadding = 0)
        {
            var rStart = -rowPadding;
            var cStart = -columnPadding;
            var rEnd = g.RowCount - f.RowCount + rowPadding + 1;
            var cEnd = g.ColumnCount - f.ColumnCount + columnPadding + 1;

            var rLength = rEnd - rStart;
            var cLength = cEnd - cStart;

            var convRows = rLength < rowStride ? rLength : (int) Math.Truncate((float) rLength / rowStride);
            var convCols = cLength < columnStride ? cLength : (int) Math.Truncate((float) cLength / columnStride);
            var conv = Matrix<float>.Build.Dense(convRows, convCols);

            for (int r = rStart; r < rEnd; r += rowStride)
            for (int c = cStart; c < cEnd; c += columnStride)
            {
                var fci = Math.Max(0, -c);
                var fri = Math.Max(0, -r);
                var gci = Math.Max(0, c);
                var gri = Math.Max(0, r);
                
                var ffci = f.ColumnCount + Math.Min(0, c);
                var ffri = f.RowCount + Math.Min(0, r);
                var ggci = g.ColumnCount + Math.Min(0, -c);
                var ggri = g.RowCount + Math.Min(0, -r);

                var cl = Math.Min(ffci, ggci);
                var rl = Math.Min(ffri, ggri);
                
                var fSub = f.SubMatrix(fri, rl, fci, cl);
                var gSub = g.SubMatrix(gri, rl, gci, cl);

                conv[(r - rStart) / rowStride, (c - cStart) / columnStride] = fSub
                    .PointwiseMultiply(gSub)
                    .ColumnSums()
                    .Sum();
            }

            return conv;
        }

        public static Matrix<float> Rotate180(this Matrix<float> matrix)
        {
            var result = Matrix<float>.Build.Dense(matrix.RowCount, matrix.ColumnCount);

            for (int c = 0; c < matrix.ColumnCount; c++)
            for (int r = 0; r < matrix.RowCount; r++)
                result[r, c] = matrix[matrix.RowCount - r - 1, matrix.ColumnCount - c - 1];

            return result;
        }
    }
}
