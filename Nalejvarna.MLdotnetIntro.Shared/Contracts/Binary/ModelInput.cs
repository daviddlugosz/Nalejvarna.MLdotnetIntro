using Microsoft.ML.Data;

namespace Nalejvarna.MLdotnetIntro.Shared.Contracts.Binary;

public class ModelInput
{
    public ModelInput(string review)
    {
        Review = review;
    }

    [LoadColumn(0)]
    [ColumnName("Review")]
    public string Review { get; set; }

    [LoadColumn(1)]
    [ColumnName("Label")]
    public bool IsPositive { get; set; }
}

