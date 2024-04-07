using Microsoft.ML.Data;

namespace Nalejvarna.MLdotnetIntro.Shared.Contracts.Multi;

public class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public int PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float[] Score { get; set; }
}

