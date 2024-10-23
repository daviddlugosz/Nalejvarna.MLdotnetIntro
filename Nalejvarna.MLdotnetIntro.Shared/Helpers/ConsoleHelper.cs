using Microsoft.ML;
using Nalejvarna.MLdotnetIntro.Shared.Extensions;

namespace Nalejvarna.MLdotnetIntro.Shared.Helpers;

public static class ConsoleHelper
{
    public static void Consume(
        PredictionEngine<Contracts.Multi.ModelInput, Contracts.Multi.ModelOutput> predictionEngine)
    {
        PrintInputPrompt();

        while (true)
        {
            var input = Console.ReadLine();
            input = input!.RemoveDiacritics();

            if (input?.ToLowerInvariant() == "quit")
            {
                break;
            }

            var sampleData = new Contracts.Multi.ModelInput(input!);
            var result = predictionEngine.Predict(sampleData);

            Console.WriteLine();
            Console.Write($"According to model your review should have rating: ");

            Console.ForegroundColor = ConsoleColor.Red;
            if (result.PredictedLabel == 0)
            {
                Console.WriteLine("odpad! (trash!)");
            }
            else
            {
                for (int i = 0; i < result.PredictedLabel; i++)
                {
                    Console.Write('*');
                }
                Console.WriteLine();
            }
            Console.ForegroundColor = ConsoleColor.White;

            Console.WriteLine();
            PrintInputPrompt();
        }
    }

    public static void Consume(
        PredictionEngine<Contracts.Binary.ModelInput, Contracts.Binary.ModelOutput> predictionEngine)
    {
        PrintInputPrompt();

        while (true)
        {
            var input = Console.ReadLine();
            input = input!.RemoveDiacritics();

            if (input?.ToLowerInvariant() == "quit")
            {
                break;
            }

            var sampleData = new Contracts.Binary.ModelInput(input!);
            var result = predictionEngine.Predict(sampleData);

            Console.WriteLine();
            Console.Write($"According to model your review is: ");

            if (result.PredictedLabel)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("positive");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("negative");

            }
            Console.ForegroundColor = ConsoleColor.White;

            Console.WriteLine();
            PrintInputPrompt();
        }
    }

    private static void PrintInputPrompt()
    {
        Console.WriteLine("Write movie review:");
        Console.WriteLine("(To quit program input 'quit')");
        Console.WriteLine();
    }
}
