using Microsoft.ML;

namespace AI.Models
{
  /// <summary>
  /// Provides methods for building, training, and saving/loading a machine learning model.
  /// </summary>
  public static class Model
  {
    /// <summary>
    /// Builds and trains a machine learning model for review classification.
    /// </summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="trainingDataFilePath">The file path of the training trainingDataView.</param>
    /// <param name="modelPath">The file path to save the trained model.</param>
    /// <returns>The trained model.</returns>
    public static ITransformer BuildAndTrain(MLContext mlContext, string trainingDataFilePath, string modelPath)
    {
      var trainingDataView = mlContext.Data.LoadFromTextFile<ReviewData>(trainingDataFilePath, separatorChar: '\t', hasHeader: true);
      var pipeline = CreatePipeline(mlContext);
      var model = pipeline.Fit(trainingDataView);

      Save(mlContext, model, modelPath, trainingDataView.Schema);

      return model;
    }

    /// <summary>
    /// Saves the trained model to a file.
    /// </summary>
    /// <param name="context">The ML context.</param>
    /// <param name="model">The trained model.</param>
    /// <param name="modelPath">The file path to save the model.</param>
    /// <param name="schema">The trainingDataView view schema.</param>
    internal static void Save(MLContext context, ITransformer model, string modelPath, DataViewSchema schema)
    {
      context.Model.Save(model, schema, modelPath);
      Console.WriteLine($"Model saved to {modelPath}");
    }

    /// <summary>
    /// Loads a trained model from a file.
    /// </summary>
    /// <param name="context">The ML context.</param>
    /// <param name="modelPath">The file path of the saved model.</param>
    /// <returns>The loaded model.</returns>
    public static ITransformer Load(MLContext context, string modelPath)
    {
      DataViewSchema schema;
      var model = context.Model.Load(modelPath, out schema);
      return model;
    }

    /// <summary>
    /// Creates the machine learning pipeline for review classification.
    /// </summary>
    /// <param name="context">The ML context.</param>
    /// <returns>The machine learning pipeline.</returns>
    public static IEstimator<ITransformer> CreatePipeline(MLContext context)
    {
      return context.Transforms.Text.FeaturizeText("Features", "ReviewText")
          .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label"))
          .AppendCacheCheckpoint(context);
    }
  }
}