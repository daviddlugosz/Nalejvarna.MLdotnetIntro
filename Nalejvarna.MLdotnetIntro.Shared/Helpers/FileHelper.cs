namespace Nalejvarna.MLdotnetIntro.Shared.Helpers;

public static class FileHelper
{
    public static (bool, string, string, string) GetAllModelPaths(string fileName)
    {
        var dataPath = $"..\\..\\..\\..\\Nalejvarna.MLdotnetIntro.Shared\\Data\\{fileName}.tsv";
        var dataFileExists = File.Exists(dataPath);

        return (dataFileExists,
            dataPath,
            $"..\\..\\..\\Model\\{fileName}_model.txt",
            $"..\\..\\..\\Model\\{fileName}_model.zip");
    }
}