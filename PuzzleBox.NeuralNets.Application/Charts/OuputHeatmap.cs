using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace PuzzleBox.TheHive.Forms.Charts
{
    public partial class OuputHeatmap : UserControl
    {
        private HeatMapSeries _heatMapSeries;
        private PlotModel _heatMapModel;
        private double[,] _headmapData;
        
        public int Resolution { get; private set; }

        public OuputHeatmap()
        {
            InitializeComponent();
            InitChart();
        }

        private void InitChart()
        {
            Resolution = 50;

            _heatMapModel = new PlotModel { Title = "Output" };

            _heatMapModel.Axes.Add(new LinearColorAxis
            {
                Palette = OxyPalettes.Gray(256),
            });

            _headmapData = new double[Resolution, Resolution];
            _heatMapSeries = new HeatMapSeries
            {
                X0 = 0,
                X1 = 1,
                Y0 = 0,
                Y1 = 1,
                Interpolate = false,
                RenderMethod = HeatMapRenderMethod.Bitmap,
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
