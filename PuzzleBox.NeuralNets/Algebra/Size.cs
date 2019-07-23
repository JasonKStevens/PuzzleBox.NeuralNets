using System;
using System.Linq;

namespace PuzzleBox.NeuralNets.Algebra
{
    public class Size : IEquatable<Size>
    {
        public int[] Dimensions { get; }
        public int Length { get; }

        public Size(params int[] dimensions)
        {
            Dimensions = dimensions.Length == 0 ? new int[] { 0 } : dimensions;
            Length = dimensions.Aggregate(1, (m, d) => d * m);
        }

        public Size Clone()
        {
            return new Size(Dimensions);
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