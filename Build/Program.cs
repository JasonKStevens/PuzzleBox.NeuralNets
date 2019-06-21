using static Bullseye.Targets;
using static SimpleExec.Command;
using static System.IO.Directory;
using static Build.BuildHelper;

namespace Build
{
    class Program
    {
        public const string Clean = "clean";
        public const string Build = "build";
        public const string Test = "test";
        public const string Pack = "pack";
        public const string Publish = "publish";

        static void Main(string[] args)
        {
            Target(Clean, () =>
            {
                DeleteDirectories(PublishDir, ArtifactsDir);
                CreateDirectory(ArtifactsDir);
                CleanSolution();
            });

            Target(Build, DependsOn(Clean), () =>
            {
                BuildSolution();
            });

            Target(Test, DependsOn(Build), () =>
            {
                RunTests("PartPay.NeuralNets.Test");
            });

            Target(Pack, DependsOn(Test), () =>
            {
                Pack("PuzzleBox.NeuralNets");
                Pack("PuzzleBox.NeuralNets.Test");
            });

            Target(Publish, DependsOn(Pack), () =>
            {
                Publish(GetNugetApiKey(), server: "https://api.nuget.org/v3/index.json");
            });

            Target("default", DependsOn(Pack));
            RunTargetsAndExit(args);
        }
    }
}
