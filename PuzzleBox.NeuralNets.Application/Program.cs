using System;
using System.Windows.Forms;

namespace PuzzleBox.TheHive.Forms
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            MathNet.Numerics.Control.UseBestProviders();

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
