using MathNet.Numerics.LinearAlgebra;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Training
{
    public class TrainingRun
    {
        private readonly Tensor[] _values;
        private readonly Tensor[] _errors;

        public int Counter { get; set; }
        public int BatchSize { get; set; }
        public float Cost { get; set; }

        public Tensor Input
        {
            get { return _values[Counter]; }
            set { _values[Counter] = value; }
        }

        public Tensor Output
        {
            get { return _values[Counter + 1]; }
            set { _values[Counter + 1] = value; }
        }

        public Tensor InputError
        {
            get { return _errors[Counter]; }
            set { _errors[Counter] = value; }
        }

        public Tensor OutputError
        {
            get { return _errors[Counter + 1]; }
            set { _errors[Counter + 1] = value; }
        }

        public Matrix<float>[] WeightsDeltas { get; private set; }

        public Matrix<float> WeightsDelta
        {
            get { return WeightsDeltas[Counter]; }
            set { WeightsDeltas[Counter] = value; }
        }

        public TrainingRun(int layerCount)
        {
            BatchSize = 0;
            Counter = 0;
            _values = new Tensor[layerCount + 1];
            _errors = new Tensor[layerCount + 1];
            WeightsDeltas = new Matrix<float>[layerCount];  // Sparse
        }

        public TrainingRun Combine(TrainingRun trainingRun)
        {
            for (int i = 0; i < trainingRun.WeightsDeltas.Length; i++)
            {
                WeightsDeltas[i] = WeightsDeltas[i] == null ? trainingRun.WeightsDeltas[i] : WeightsDeltas[i] + trainingRun.WeightsDeltas[i];
            }

            BatchSize += trainingRun.BatchSize;
            Cost += trainingRun.Cost;
            return this;
        }
    }
}
