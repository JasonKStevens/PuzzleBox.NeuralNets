using System;
using System.Linq;

namespace PuzzleBox.NeuralNets.Algebra
{
    public class Size : IEquatable<Size>
    {
        public int[] Dimensions { get; }
        public int Length { get; }
        public int KernelCount { get; }

        public int TotalLength => KernelCount * Length;

        public Size(params int[] dimensions)
        {
            Dimensions = dimensions.Length == 0 ? new int[] { 0 } : dimensions;
            Length = dimensions.Aggregate(1, (m, d) => d * m);
            KernelCount = 1;
        }

        public Size(int[] dimensions, int kernelCount = 1) : this(dimensions)
        {
            KernelCount = kernelCount;
        }

        public Size Clone()
        {
            return new Size(Dimensions, KernelCount);
        }

        public override string ToString()
        {
            return $"({string.Join(", ", Dimensions)})";
        }

        public bool Equals(Size other)
        {
            if (other == null)
                return false;

            return
                Length == other.Length &&
                Dimensions.SequenceEqual(other.Dimensions);
        }
    }
}