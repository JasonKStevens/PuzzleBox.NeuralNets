using System.Windows.Forms;
using OxyPlot.Axes;
using OxyPlot.Series;
using OxyPlot;
using System;

namespace PuzzleBox.TheHive.Forms.Charts
{
    public partial class CostChart : UserControl
    {
        private PlotModel _plotModel;
        private LineSeries _costSeries;
        private LinearAxis _bottomAxes;
        private LinearAxis _leftAxes;

        public CostChart()
        {
            InitializeComponent();
            InitChart();
        }

        private void InitChart()
        {
            _plotModel = new PlotModel
            {
                Title = "Cost",
                PlotType = PlotType.Cartesian
            };

            _costSeries = new LineSeries();
            _plotModel.Series.Add(_costSeries);

            _bottomAxes = new LinearAxis { Position = AxisPosition.Bottom, AbsoluteMinimum = 0, AbsoluteMaximum = 100, Minimum = 0, Maximum = 100, Title = "Epoch" };
            _plotModel.Axes.Add(_bottomAxes);
            _leftAxes = new LinearAxis { Position = AxisPosition.Left, AbsoluteMinimum = 0, AbsoluteMaximum = 1 };
            _plotModel.Axes.Add(_leftAxes);

            accuracyPlot.Model = _plotModel;
        }

        public void Reset(int totalIterations)
        {
            _bottomAxes.AbsoluteMaximum = totalIterations;
            _bottomAxes.Maximum = totalIterations;
            _leftAxes.AbsoluteMaximum = 10;
            _plotModel.ResetAllAxes();
            _costSeries.Points.Clear();
        }

        public void AddCost(int epoch, float cost)
        {
            _costSeries.Points.Add(new DataPoint(epoch, cost));
            _leftAxes.AbsoluteMaximum = Math.Max(_leftAxes.AbsoluteMaximum, cost);
        }

        public void Draw()
        {
            _plotModel.InvalidatePlot(false);
        }
    }
}
