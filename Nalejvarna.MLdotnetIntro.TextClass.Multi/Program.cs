using Microsoft.ML;
using Nalejvarna.MLdotnetIntro.Shared.Contracts.Multi;
using Nalejvarna.MLdotnetIntro.Shared.Helpers;
using System.Text;

const string fileName = "csfd";
var (fileExists, dataFilePath, modelMetricsFilePath, modelFilePath) = FileHelper.GetAllModelPaths(fileName);

if (!fileExists)
{
    Console.WriteLine($"Soubor se vstupními daty {dataFilePath} nenalezen");
    return;
}

var mlContext = new MLContext();

if (!File.Exists(modelFilePath))
{
    var (model, testData) = BuildAndTrainModel(mlContext, dataFilePath);

    EvaluateAndSaveModel(mlContext, model, testData, modelFilePath, modelMetricsFilePath);
}

var predictionEngine = LoadModelAndCreatePredictionEngine(mlContext, modelFilePath);

ConsoleHelper.Consume(predictionEngine);

(ITransformer, IDataView) BuildAndTrainModel(
    MLContext mlContext, string dataFilePath)
{
    Console.WriteLine($"{DateTime.Now:T}\tNačítání dat...");
    Console.WriteLine();

    var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
        dataFilePath,
        separatorChar: '\t',
        hasHeader: false
    );

    var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1);
    var trainData = dataSplit.TrainSet;
    var testData = dataSplit.TestSet;

    var pipeline = mlContext.Transforms.Conversion
        .MapValueToKey("Label", "Label")
        .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", "Review"))
        .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
        .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"));

    var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

    var trainingPipeline = pipeline.Append(trainer);

    Console.WriteLine($"{DateTime.Now:T}\tStart učení modelu...");
    Console.WriteLine();

    return (trainingPipeline.Fit(trainData), testData);
}

void EvaluateAndSaveModel(MLContext mlContext,
    ITransformer model,
    IDataView testData,
    string modelFilePath,
    string modelMetricsFilePath)
{
    var transformedTest = model.Transform(testData);

    Console.WriteLine($"{DateTime.Now:T}\tStarting evaluation of the model...");
    Console.WriteLine();

    var metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

    Console.WriteLine($"{DateTime.Now:T}\tVýsledek měření:");

    var sb = new StringBuilder();
    sb.AppendLine($"Macro accuracy: {metrics.MacroAccuracy}");
    sb.AppendLine($"Micro accuracy: {metrics.MicroAccuracy}");
    // další metriky...

    sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    var metricsString = sb.ToString();
    Console.WriteLine(metricsString); ;
    File.WriteAllText(modelMetricsFilePath, metricsString);
    Console.WriteLine();

    Console.WriteLine($"{DateTime.Now:T}\tUkládání modelu...");
    Console.WriteLine();
    mlContext.Model.Save(model, testData.Schema, modelFilePath);
}

PredictionEngine<ModelInput, ModelOutput> LoadModelAndCreatePredictionEngine(MLContext mlContext, string modelFilePath)
{
    Console.WriteLine($"{DateTime.Now:T}\tNačítání modelu a vytváření predikčního engine...");
    Console.WriteLine();

    var model = mlContext.Model.Load(modelFilePath, out _);

    Console.WriteLine($"{DateTime.Now:T}\tModel načten a predikční engine vytvořen...");
    Console.WriteLine();

    return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
}
