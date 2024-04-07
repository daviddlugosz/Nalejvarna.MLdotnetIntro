namespace Nalejvarna.MLdotnetIntro.Shared.Extensions;

public static class StringExtension
{
    public static string RemoveDiacritics(this string str) => str
        .Replace("ě", "e")
        .Replace("š", "s")
        .Replace("č", "c")
        .Replace("ř", "r")
        .Replace("ž", "z")
        .Replace("ý", "y")
        .Replace("á", "a")
        .Replace("í", "i")
        .Replace("é", "e")
        .Replace("ú", "u")
        .Replace("ů", "u")
        .Replace("ť", "t")
        .Replace("ó", "o")
        .Replace("ď", "d")
        .Replace("ň", "n")
        .Replace("Ě", "E")
        .Replace("Š", "S")
        .Replace("Č", "C")
        .Replace("Ř", "R")
        .Replace("Ž", "Z")
        .Replace("Ý", "Y")
        .Replace("Á", "A")
        .Replace("Í", "I")
        .Replace("É", "E")
        .Replace("Ú", "U")
        .Replace("Ť", "T")
        .Replace("Ó", "O")
        .Replace("Ď", "D")
        .Replace("Ň", "N");
}
