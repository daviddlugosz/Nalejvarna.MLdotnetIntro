namespace Nalejvarna.MLdotnetIntro.Shared.Helpers;

public static class FileHelper
{
    public static (bool, string, string, string) GetAllModelPaths(string fileName)
    {
        var modelPath = $"..\\..\\..\\..\\Nalejvarna.MLdotnetIntro.Shared\\Data\\{fileName}.tsv";
        var modelExists = File.Exists(modelPath);

        return (modelExists,
            modelPath,
            $"..\\..\\..\\Model\\{fileName}_model.txt",
            $"..\\..\\..\\Model\\{fileName}_model.zip");
    }
}