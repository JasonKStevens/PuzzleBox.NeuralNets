using MathNet.Numerics.LinearAlgebra;

namespace PuzzleBox.NeuralNets.CostFunctions
{
    /// <summary>
    /// The cost function used by the trainer on the last layer when backpropogating the error.
    /// </summary>
    public interface ICostFunction
    {
        /// <summary>
        /// Calulates the cost of the actual output against the desired output.
        /// </summary>
        /// <param name="h">Output values or hypothesis.</param>
        /// <param name="y">The desired output (labeled) values.</param>
        Vector<float> CalcCost(Vector<float> h, Vector<float> y);

        /// <summary>
        /// Calulates the cost gradient of the actual output with respect to the desired output.
        /// </summary>
        /// <param name="h">Output values or hypothesis.</param>
        /// <param name="y">The desired output (labeled) values.</param>
        Vector<float> CalcCostGradient(Vector<float> h, Vector<float> y);
    }
}