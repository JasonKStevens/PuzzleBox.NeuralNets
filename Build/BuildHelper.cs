using System;
using System.IO;
using System.Linq;
using static System.IO.Directory;
using static SimpleExec.Command;

namespace Build
{
    internal static class BuildHelper
    {
        public const string PublishDir = "publish";
        public const string ArtifactsDir = "artifacts";

        public const string NugetServer = "https://";

        public static void CleanSolution(string configuration = "Release")
        {
            Run("dotnet", $"clean --configuration={configuration}");
        }

        public static void BuildSolution(string configuration = "Release")
        {
            Run("dotnet", $"clean --configuration={configuration}");
        }

        public static void RunTests(string project, string configuration = "Release")
        {
            Run("dotnet", $"test {Path.Combine(project, project)}.csproj --configuration={configuration} --no-build");
        }

        public static void Pack(string project, string artifactsDirectory = ArtifactsDir)
        {
            Run("dotnet", $"pack {project}/{project}.csproj -c Release -o ../{artifactsDirectory} --no-build");
        }

        public static void Publish(
            string apiKey,
            string artifactsDirectory = ArtifactsDir,
            string server = NugetServer)
        {
            var packagesToPush = GetFiles(artifactsDirectory, "*.nupkg", SearchOption.TopDirectoryOnly)
                .Select(f => new FileInfo(f))
                .ToList();

            Console.WriteLine($"Packages found to publish: {string.Join("; ", packagesToPush)}");

            foreach (var packageToPush in packagesToPush)
            {
                Run("dotnet", $"nuget push {packageToPush.FullName} -s {server} -k {apiKey}");
            }
        }

        public static void DeleteDirectories(params string[] directories)
        {
            foreach (var directory in directories)
            {
                if (Exists(directory))
                {
                    Delete(directory);
                }
            }
        }

        public static string GetNugetApiKey()
        {
            var apiKey = Environment.GetEnvironmentVariable("PuzzleBox.NeuralNets.ApiKey");

            if (string.IsNullOrWhiteSpace(apiKey))
            {
                throw new Exception("Nuget API key should be set in environment variable 'PuzzleBox.NeuralNets.ApiKey'. Packages will not be pushed.");
            }

            return apiKey;
        }
    }
}
