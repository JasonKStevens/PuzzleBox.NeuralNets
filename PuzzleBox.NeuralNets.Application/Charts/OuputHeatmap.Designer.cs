namespace PuzzleBox.TheHive.Forms.Charts
{
    partial class OuputHeatmap
    {
        /// <summary> 
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.heatMap = new OxyPlot.WindowsForms.PlotView();
            this.SuspendLayout();
            // 
            // heatMap
            // 
            this.heatMap.Dock = System.Windows.Forms.DockStyle.Fill;
            this.heatMap.Location = new System.Drawing.Point(0, 0);
            this.heatMap.Name = "heatMap";
            this.heatMap.PanCursor = System.Windows.Forms.Cursors.Hand;
            this.heatMap.Size = new System.Drawing.Size(150, 150);
            this.heatMap.TabIndex = 5;
            this.heatMap.Text = "plotView1";
            this.heatMap.ZoomHorizontalCursor = System.Windows.Forms.Cursors.SizeWE;
            this.heatMap.ZoomRectangleCursor = System.Windows.Forms.Cursors.SizeNWSE;
            this.heatMap.ZoomVerticalCursor = System.Windows.Forms.Cursors.SizeNS;
            // 
            // OuputHeatmap
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.heatMap);
            this.Name = "OuputHeatmap";
            this.ResumeLayout(false);

        }

        #endregion

        private OxyPlot.WindowsForms.PlotView heatMap;
    }
}
