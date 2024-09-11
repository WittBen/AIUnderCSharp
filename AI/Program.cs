using AI.Models;
using AI.Files;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AI
{
  public class Program
  {
    static void Main(string[] args)
    {
      
      //I use the files from the project here so that they can also be filled here via visual studio
      // this is only optional
      string baseDirectory = @"H:\_Artikel\Tech-Specific\C#\AI under CSharp\AI\AI"; 
      string subDirectory = "TrainingData";

      string trainingDataFilePath = Path.Combine(baseDirectory, subDirectory, "Reviews.tsv");
      string newReviewsFilePath = Path.Combine(baseDirectory, subDirectory, "NewReviews.tsv");
      //##############################################################################################

      string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.zip");

      MLContext mlContext = new MLContext();

      if (!System.IO.File.Exists(modelPath))
      {
        Model.BuildAndTrain(mlContext, trainingDataFilePath, modelPath);
      }


      Console.WriteLine("Would you like to enter new ratings (1) or classify existing ratings (2)?");

      if (int.TryParse(Console.ReadLine(), out int choice) && (choice == 1 || choice == 2))
      {
        if (choice == 1)
        {
          ReviewFile.EnterNewReviews(mlContext, trainingDataFilePath, modelPath, newReviewsFilePath);
        }
        else
        {
          ReviewFile.ClassifyNewReviews(mlContext, trainingDataFilePath, modelPath, newReviewsFilePath);
        }
      }
      else
      {
        Console.WriteLine("Invalid input. The program is terminated.");
      }
    }
  }

  public class ReviewData
  {
    [LoadColumn(0)]
    public string ReviewText { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
  }

  public class ReviewPrediction
  {
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
  }
}