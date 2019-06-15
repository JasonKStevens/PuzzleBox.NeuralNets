using NUnit.Framework;
using MathNet.Numerics;

namespace PuzzleBox.NeuralNets.Test
{
    [SetUpFixture]
    public class Setup
    {
        [OneTimeSetUp]
        public void GlobalSetup()
        {
            Control.UseBestProviders();
        }
    }
}
