using AI.Models;
using AI;
using Microsoft.ML;
using System.Text;

namespace AI.Files
{
  /// <summary>
  /// Provides methods for managing review data files and classifying new reviews.
  /// </summary>
  public static class ReviewFile
  {
    /// <summary>
    /// Allows the user to enter new reviews and saves them to a file.
    /// </summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="newReviewsFilePath">The file path to save the new reviews.</param>
    public static void EnterNewReviews(MLContext mlContext, string trainingDataFilePath, string modelPath, string newReviewsFilePath)
    {
      List<ReviewData> newReviews = new List<ReviewData>();

      ITransformer model;
      if (File.Exists(modelPath))
      {
        model = Model.Load(mlContext, modelPath);
      }
      else
      {
        Console.WriteLine("The model was not found. No classification can be performed.");

        return;
      }

      var predictionEngine = mlContext.Model.CreatePredictionEngine<ReviewData, ReviewPrediction>(model);

      // User1 enters ratings
      while (true)
      {
        Console.WriteLine("please enter a rating (or 'exit' to exit):");
        string inputReviewText = Console.ReadLine();
        if (inputReviewText.ToLower() == "exit")
          break;

        var input = new ReviewData { ReviewText = inputReviewText };
        newReviews.Add(input);

        // Classify the new rating and display the result
        var prediction = predictionEngine.Predict(input);
        Console.WriteLine($"The classification for the rating ‘{ input.ReviewText}’ is: { (prediction.PredictedLabel ? "Positiv" : "Negativ")}");
      }

      // Write ratings in a separate file
      WriteReviewsToFile(newReviews, newReviewsFilePath, trainingDataFilePath);
      Console.WriteLine("All new ratings have been saved.");
    }


    /// <summary>
    /// Writes new reviews to the training data file and removes them from the new reviews file.
    /// </summary>
    /// <param name="newReviewsFilePath">The file path of the new reviews.</param>
    /// <param name="trainingDataFilePath">The file path of the training data.</param>
    public static void WriteReviewsToFile(List<ReviewData> reviews, string newReviewsFilePath, string trainingDataFilePath)
    {
      try
      {
        var sb = new StringBuilder();

        foreach (var review in reviews)
        {
          sb.AppendLine(review.ReviewText);
        }

        File.WriteAllText(newReviewsFilePath, sb.ToString());
      }
        catch (Exception ex)
        {
          Console.WriteLine($"Error updating training data: {ex.Message}");
        }
      }

    /// <summary>
    /// Classifies new reviews using a trained model and updates the training data if necessary.
    /// </summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="trainingDataFilePath">The file path of the training data.</param>
    /// <param name="modelPath">The file path of the trained model.</param>
    /// <param name="newReviewsFilePath">The file path of the new reviews to classify.</param>
    public static void ClassifyNewReviews(MLContext mlContext, string trainingDataFilePath, string modelPath, string newReviewsFilePath)
    {
      ITransformer model;
      if (File.Exists(modelPath))
      {
        model = Model.Load(mlContext, modelPath);
      }
      else
      {
        Console.WriteLine("The model was not found. Please train a model first.");
        return;
      }

      var predictionEngine = mlContext.Model.CreatePredictionEngine<ReviewData, ReviewPrediction>(model);

      // Read new reviews from the separate file
      var newReviews = File.ReadAllLines(newReviewsFilePath).ToList();

      // Create a new list for reviews that have not been classified yet
      var remainingReviews = new List<string>();

      // Classify only new reviews
      var existingReviewsSet = new HashSet<string>(File.ReadAllLines(trainingDataFilePath));
      foreach (var reviewText in newReviews)
      {
        // Check if the review already exists in the training data
        if (existingReviewsSet.Contains(reviewText))
        {
          Console.WriteLine($"The review '{reviewText}' already exists in the training data and will be ignored.");
          continue;
        }

        // Classify the review
        var input = new ReviewData { ReviewText = reviewText };
        var prediction = predictionEngine.Predict(input);
        Console.WriteLine($"The classification for the review '{input.ReviewText}' is: {(prediction.PredictedLabel ? "Positive" : "Negative")}");

        // Check if the classification is correct
        Console.WriteLine("Is the classification correct? (y/n)");
        var feedback = Console.ReadLine();
        bool correctClassification = feedback.ToLower() == "y";

        if (!correctClassification)
        {
          // User2 enters the correct classification
          Console.WriteLine("Please enter the correct classification (0 for negative, 1 for positive):");
          if (int.TryParse(Console.ReadLine(), out int correctLabel) && (correctLabel == 0 || correctLabel == 1))
          {
            // Add a new line to the TSV file
            using (var writer = new StreamWriter(trainingDataFilePath, true, Encoding.UTF8))
            {
              writer.WriteLine($"{reviewText}\t{correctLabel}");
            }
          }
          else
          {
            Console.WriteLine("Invalid input. The classification has not been updated.");
          }
        }
        else
        {
          // Add the review to the training data
          using (var writer = new StreamWriter(trainingDataFilePath, true, Encoding.UTF8))
          {
            writer.WriteLine($"{reviewText}\t{Convert.ToInt32(prediction.PredictedLabel)}");
          }
        }
      }

      // Create an empty list for the NewReviews file
      var emptyList = new List<string>();

      // Clear the contents of the NewReviews file
      File.WriteAllLines(newReviewsFilePath, emptyList);

      // Retrain and save the model
      model = Model.BuildAndTrain(mlContext, trainingDataFilePath, modelPath);
    }
  }
}