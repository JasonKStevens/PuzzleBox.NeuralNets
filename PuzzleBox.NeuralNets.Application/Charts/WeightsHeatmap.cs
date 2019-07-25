using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace PuzzleBox.TheHive.Forms.Charts
{
    public partial class WeightsHeatmap : UserControl
    {
        private HeatMapSeries _heatMapSeries;
        private PlotModel _heatMapModel;
        private double[,] _headmapData;
        
        public WeightsHeatmap()
        {
            InitializeComponent();
            InitChart();
        }

        private void InitChart()
        {
            _heatMapModel = new PlotModel
            {
                Title = "Weights"
            };

            _heatMapModel.Axes.Add(new LinearColorAxis
            {
                Palette = OxyPalettes.Gray(256),
            });

            _headmapData = new double[15, 15];
            _heatMapSeries = new HeatMapSeries
            {
                X0 = 0,
                X1 = 9,
                Y0 = 0,
                Y1 = 9,
                Interpolate = false,
                Data = _headmapData
            };

            _heatMapModel.Series.Add(_heatMapSeries);
            heatMap.Model = _heatMapModel;
        }

        public void SetPoint(int x, int y, float value)
        {
            _headmapData[x, y] = value;
        }

        public void Draw()
        {
            _heatMapModel.InvalidatePlot(true);
        }
    }
}
