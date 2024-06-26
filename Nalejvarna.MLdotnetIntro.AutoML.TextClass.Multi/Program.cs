﻿using Microsoft.ML;
using Microsoft.ML.AutoML;
using System.Data;
using System.Text;
using Nalejvarna.MLdotnetIntro.Shared.Contracts.Multi;
using Nalejvarna.MLdotnetIntro.Shared.Helpers;

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
    var (model, testData) = BuildAndTrainModel(mlContext, dataFilePath, modelMetricsFilePath);

    EvaluateAndSaveModel(mlContext, model, testData, modelFilePath, modelMetricsFilePath);
}

var predictionEngine = LoadModelAndCreatePredictionEngine(mlContext, modelFilePath);

ConsoleHelper.Consume(predictionEngine);

(ITransformer, IDataView) BuildAndTrainModel(
    MLContext mlContext, string dataFilePath, string modelMetricsFilePath)
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

    var experimentSettings = new MulticlassExperimentSettings
    {
        MaxExperimentTimeInSeconds = 1800,
        OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
    };

    Console.WriteLine($"{DateTime.Now:T}\tSpouštění experimentu pro nalezení nejvhodnějíšho algoritmu pro vytvoření modelu...");

    var experiment = mlContext
        .Auto()
        .CreateMulticlassClassificationExperiment(experimentSettings)
        .Execute(trainData);

    var sb = new StringBuilder();
    sb.AppendLine();
    sb.AppendLine($"Nejlepší nalezení algoritmus: {experiment.BestRun.TrainerName}, MacroAccuracy: {experiment.BestRun.ValidationMetrics.MacroAccuracy}");
    sb.AppendLine();
    foreach (var trainer in experiment.RunDetails
        .Where(rd => rd.TrainerName != experiment.BestRun.TrainerName))
    {
        sb.AppendLine($"{trainer.TrainerName} MacroAccuracy: {trainer.ValidationMetrics.MacroAccuracy}");
    };
    var experimentString = sb.ToString();
    Console.WriteLine(experimentString);
    File.WriteAllText(modelMetricsFilePath, experimentString);
    Console.WriteLine();

    return (experiment.BestRun.Model, testData);
}

void EvaluateAndSaveModel(MLContext mlContext,
    ITransformer model,
    IDataView testData,
    string modelFilePath, 
    string modelMetricsFilePath)
{
    var transformedTest = model.Transform(testData);

    Console.WriteLine($"{DateTime.Now:T}\tSpouštění měření modelu...");
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
    File.AppendAllText(modelMetricsFilePath, metricsString);
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
