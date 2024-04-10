using Microsoft.ML.Data;

namespace Nalejvarna.MLdotnetIntro.Shared.Contracts.Binary;

public class ModelOutput
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }
}

