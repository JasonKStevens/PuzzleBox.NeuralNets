using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.NeuralNets.Layers
{
    public interface ILayer
    {
        Size InputSize { get; }
        Size OutputSize { get; }

        Tensor FeedForwards(Tensor input);
        void BackPropagate(TrainingRun trainingRun);
    }
}
