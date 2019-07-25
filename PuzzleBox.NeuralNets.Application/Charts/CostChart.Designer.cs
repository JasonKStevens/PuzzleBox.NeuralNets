namespace PuzzleBox.TheHive.Forms.Charts
{
    partial class CostChart
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
            this.accuracyPlot = new OxyPlot.WindowsForms.PlotView();
            this.SuspendLayout();
            // 
            // accuracyPlot
            // 
            this.accuracyPlot.Dock = System.Windows.Forms.DockStyle.Fill;
            this.accuracyPlot.Location = new System.Drawing.Point(0, 0);
            this.accuracyPlot.Name = "accuracyPlot";
            this.accuracyPlot.PanCursor = System.Windows.Forms.Cursors.Hand;
            this.accuracyPlot.Size = new System.Drawing.Size(150, 150);
            this.accuracyPlot.TabIndex = 4;
            this.accuracyPlot.Text = "plotView1";
            this.accuracyPlot.ZoomHorizontalCursor = System.Windows.Forms.Cursors.SizeWE;
            this.accuracyPlot.ZoomRectangleCursor = System.Windows.Forms.Cursors.SizeNWSE;
            this.accuracyPlot.ZoomVerticalCursor = System.Windows.Forms.Cursors.SizeNS;
            // 
            // AccuracySeries
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.accuracyPlot);
            this.Name = "AccuracySeries";
            this.ResumeLayout(false);

        }

        #endregion

        private OxyPlot.WindowsForms.PlotView accuracyPlot;
    }
}
