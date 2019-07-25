namespace PuzzleBox.TheHive.Forms
{
    partial class Form1
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

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.startTrainingBtn = new System.Windows.Forms.Button();
            this.weightsGrid = new System.Windows.Forms.DataGridView();
            this.accuracyChart = new PuzzleBox.TheHive.Forms.Charts.CostChart();
            this.ouputHeatmap = new PuzzleBox.TheHive.Forms.Charts.OuputHeatmap();
            this.weightsHeatmap = new PuzzleBox.TheHive.Forms.Charts.WeightsHeatmap();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.weightsGrid)).BeginInit();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.FixedPanel = System.Windows.Forms.FixedPanel.Panel2;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.tableLayoutPanel1);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.startTrainingBtn);
            this.splitContainer1.Size = new System.Drawing.Size(636, 554);
            this.splitContainer1.SplitterDistance = 508;
            this.splitContainer1.TabIndex = 4;
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 1;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Controls.Add(this.accuracyChart, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.ouputHeatmap, 0, 1);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(508, 554);
            this.tableLayoutPanel1.TabIndex = 3;
            // 
            // startTrainingBtn
            // 
            this.startTrainingBtn.Location = new System.Drawing.Point(16, 54);
            this.startTrainingBtn.Name = "startTrainingBtn";
            this.startTrainingBtn.Size = new System.Drawing.Size(127, 45);
            this.startTrainingBtn.TabIndex = 3;
            this.startTrainingBtn.Text = "Train";
            this.startTrainingBtn.UseVisualStyleBackColor = true;
            this.startTrainingBtn.Click += new System.EventHandler(this.startTrainingBtn_Click);
            // 
            // weightsGrid
            // 
            this.weightsGrid.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.weightsGrid.Dock = System.Windows.Forms.DockStyle.Fill;
            this.weightsGrid.Location = new System.Drawing.Point(405, 294);
            this.weightsGrid.Name = "weightsGrid";
            this.weightsGrid.RowHeadersWidth = 62;
            this.weightsGrid.RowTemplate.Height = 28;
            this.weightsGrid.Size = new System.Drawing.Size(397, 285);
            this.weightsGrid.TabIndex = 11;
            // 
            // accuracyChart
            // 
            this.accuracyChart.Dock = System.Windows.Forms.DockStyle.Fill;
            this.accuracyChart.Location = new System.Drawing.Point(3, 3);
            this.accuracyChart.Name = "accuracyChart";
            this.accuracyChart.Size = new System.Drawing.Size(502, 271);
            this.accuracyChart.TabIndex = 6;
            // 
            // ouputHeatmap
            // 
            this.ouputHeatmap.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ouputHeatmap.Location = new System.Drawing.Point(3, 280);
            this.ouputHeatmap.Name = "ouputHeatmap";
            this.ouputHeatmap.Size = new System.Drawing.Size(502, 271);
            this.ouputHeatmap.TabIndex = 7;
            // 
            // weightsHeatmap
            // 
            this.weightsHeatmap.Dock = System.Windows.Forms.DockStyle.Fill;
            this.weightsHeatmap.Location = new System.Drawing.Point(405, 3);
            this.weightsHeatmap.Name = "weightsHeatmap";
            this.weightsHeatmap.Size = new System.Drawing.Size(397, 285);
            this.weightsHeatmap.TabIndex = 10;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(636, 554);
            this.Controls.Add(this.splitContainer1);
            this.Name = "Form1";
            this.Text = "NeuralNets";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.tableLayoutPanel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.weightsGrid)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.Button startTrainingBtn;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private Charts.CostChart accuracyChart;
        private Charts.OuputHeatmap ouputHeatmap;
        private Charts.WeightsHeatmap weightsHeatmap;
        private System.Windows.Forms.DataGridView weightsGrid;
    }
}

