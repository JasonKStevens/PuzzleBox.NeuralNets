using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;

namespace PuzzleBox.NeuralNets.Training
{
    public class Trainer
    {
        private readonly Net _net;
        private readonly float _learningRate;
        private ICostFunction _costFunction;

        public Trainer(Net net, float learningRate = 0.1f)
        {
            _net = net;
            _learningRate = learningRate;
            _costFunction = new QuadraticCost();
        }

        public Task<float> TrainAsync(
            int epochSize,
            IEnumerable<(Tensor input, Tensor output)> batch,
            CancellationToken? token = null)
        {
            var taskCompletionSource = new TaskCompletionSource<float>();
            float latestCost = 0;

            Train(epochSize, batch, token)
                .Subscribe(
                    cost => latestCost = cost,
                    error => taskCompletionSource.SetException(error),
                    () => taskCompletionSource.SetResult(latestCost)
                );

            return taskCompletionSource.Task;
        }

        internal void SetCostFunction(ICostFunction costFunction)
        {
            _costFunction = costFunction;
        }

        public IObservable<float> Train(
            int epochSize,
            IEnumerable<(Tensor input, Tensor output)> batchData,
            CancellationToken? token = null)
        {
            var subject = new Subject<float>();

            Task.Run(() =>
            {
                for (int i = 0; i < epochSize; i++)
                {
                    if (token?.IsCancellationRequested == true)
                        break;
                    
                    TrainingRun batchRun;

                    try
                    {
                        batchRun = TrainBatch(batchData);
                    }
                    catch (Exception ex)
                    {
                        subject.OnError(ex);
                        return;
                    }

                    subject.OnNext(batchRun.Cost / batchRun.BatchSize);
                }

                subject.OnCompleted();
            });

            return subject;
        }

        private TrainingRun TrainBatch(IEnumerable<(Tensor input, Tensor output)> batchData)
        {
            TrainingRun batchRun = batchData.AsParallel()
                .Select(b => Train(b.input, b.output))
                .Aggregate(new TrainingRun(_net.Layers.Count), (m, t) => m.Combine(t));

            _net.ApplyTraining(batchRun, _learningRate);
            return batchRun;
        }

        public TrainingRun Train(Tensor input, Tensor output)
        {
            var trainingRun = _net.FeedForwardsTraining(input);

            trainingRun.Cost = _costFunction.CalcCost(trainingRun.Output, output).Sum();
            trainingRun.OutputError = _costFunction.CalcCostGradient(trainingRun.Output, output);
            _net.BackPropagate(trainingRun);

            return trainingRun;
        }
    }
}
