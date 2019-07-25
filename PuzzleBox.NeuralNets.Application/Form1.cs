using System;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using PuzzleBox.NeuralNets;
using PuzzleBox.NeuralNets.Algebra;
using PuzzleBox.NeuralNets.CostFunctions;
using PuzzleBox.NeuralNets.Layers;
using PuzzleBox.NeuralNets.Layers.Activations;
using PuzzleBox.NeuralNets.Layers.Weighted;
using PuzzleBox.NeuralNets.Training;

namespace PuzzleBox.TheHive.Forms
{
    public partial class Form1 : Form
    {
        private IDisposable _trainingSub;
        private CancellationTokenSource _trainingCancellation;

        public Form1()
        {
            InitializeComponent();
            Application.Idle += HandleApplicationIdle;
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            Application.Idle -= HandleApplicationIdle;
            CancelTraining();
        }

        private void HandleApplicationIdle(object sender, EventArgs e)
        {
            accuracyChart.Draw();
            ouputHeatmap.Draw();
            weightsHeatmap.Draw();
        }

        private void CancelTraining()
        {
            _trainingCancellation?.Cancel();
            _trainingCancellation?.Dispose();

            _trainingSub?.Dispose();
        }

        private void startTrainingBtn_Click(object sender, EventArgs e)
        {
            var net = new Net(2)
                .Dense(4)
                .Relu()
                .Dense(1)
                .Sigmoid();

            var data = new (Tensor input, Tensor output)[]
            {
                (new float[] { 0, 0 }, 0),
                (new float[] { 0, 0.25f }, 0),
                (new float[] { 0, 0.5f }, 0),
                (new float[] { 0, 0.75f }, 0),
                (new float[] { 0, 1 }, 0),
                (new float[] { 0.25f, 0 }, 0),
                (new float[] { 0.5f, 0 }, 0),
                (new float[] { 0.75f, 0 }, 0),
                (new float[] { 0.25f, 0.25f }, 0.5f),
                (new float[] { 0.25f, 0.75f }, 0.5f),
                (new float[] { 0.75f, 0.25f }, 0.5f),
                (new float[] { 0.25f, 0.5f }, 1),
                (new float[] { 0.5f, 0.5f }, 1),
                (new float[] { 0.75f, 0.5f }, 1),
                (new float[] { 0.5f, 0.25f }, 1),
                (new float[] { 0.5f, 0.5f }, 1),
                (new float[] { 0.5f, 0.75f }, 1),
                (new float[] { 0.75f, 0.75f }, 0.5f),
                (new float[] { 0.25f, 1 }, 0),
                (new float[] { 0.5f, 1 }, 0),
                (new float[] { 0.75f, 1 }, 0),
                (new float[] { 1, 0 }, 0),
                (new float[] { 1, 0.25f }, 0),
                (new float[] { 1, 0.5f }, 0),
                (new float[] { 1, 0.75f }, 0),
                (new float[] { 1, 1 }, 0),
            };

            var trainer = new Trainer(net, 0.15f)
                .UseQuadraticCost();
            int epoch = 0;
            int totalEpochs = 100000;

            CancelTraining();
            accuracyChart.Reset(totalEpochs);

            weightsGrid.ColumnCount = 20;
            weightsGrid.RowCount = 20;

            _trainingCancellation = new CancellationTokenSource();
            _trainingSub = trainer.Train(totalEpochs, data, _trainingCancellation.Token)
                .Subscribe(
                    cost =>
                    {
                        var res = ouputHeatmap.Resolution;
                        accuracyChart.AddCost(epoch++, 100 * cost);

                        if (epoch % 1000 > 0)
                            return;

                        for (int x = 0; x < res; x++)
                        for (int y = 0; y < res; y++)
                        {
                            var output = net.FeedForwards((float)x / res, (float)y / res);
                            ouputHeatmap.SetPoint(x, y, output[0]);
                        }

                        var weightLayers = net.Layers.OfType<IHaveWeights>().ToArray();
                        var baseColumn = 0;

                        for (int i = 0; i < weightLayers.Length; i++)
                        {
                            var weights = weightLayers[i].GetWeights();

                            for (int r = 0; r < weights.RowCount; r++)
                            {
                                for (int c = 0; c < weights.ColumnCount; c++)
                                {
                                    var weight = weights[r, c];
                                    weightsHeatmap.SetPoint(baseColumn + c, r, weight);
                                }

                                var rowValues = weights.Row(r).Select(v => v.ToString()).ToArray();
                                //weightsGrid.Rows[baseColumn + r].SetValues(rowValues);
                            }

                            baseColumn += weights.ColumnCount + 1;
                        }
                    },
                    ex =>
                    {
                        if (ex is AggregateException aggEx)
                            ex = aggEx.Flatten();
                        while (ex.InnerException != null) ex = ex.InnerException;
                        MessageBox.Show(ex.Message, "Training Error");
                    }
                );
        }
    }
}
