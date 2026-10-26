using LightGBMMultiClassClassifier.Data;
using Microsoft.ML;

var mlContext = new MLContext(seed: 42);
string csvPath = Path.Combine(AppContext.BaseDirectory, "tickets.csv");
var data = mlContext.Data.LoadFromTextFile<TicketData>(path: csvPath, hasHeader: true,separatorChar: ',');

var split = mlContext.Data.TrainTestSplit(data,testFraction: 0.20,seed: 42);

Console.WriteLine($"Training records: {split.TrainSet.GetRowCount()}");
Console.WriteLine($"Test records: {split.TestSet.GetRowCount()}");

var pipeline =
    mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "Label",
        inputColumnName: nameof(TicketData.Category))
    .Append(
        mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(TicketData.TicketText)))
    .Append(
        mlContext.MulticlassClassification.Trainers.LightGbm(
            labelColumnName: "Label",
            featureColumnName: "Features"))
    .Append(
        mlContext.Transforms.Conversion.MapKeyToValue(
            outputColumnName: "PredictedLabel"));

// STEP 4: Train the model.
var model = pipeline.Fit(split.TrainSet);

Console.WriteLine("\nLightGBM training completed.");

// STEP 5: Generate predictions for test data.
var testPredictions = model.Transform(split.TestSet);

// STEP 6: Evaluate the model.
var metrics = mlContext.MulticlassClassification.Evaluate(
    testPredictions,
    labelColumnName: "Label",
    predictedLabelColumnName: "PredictedLabel");

Console.WriteLine("\n=== Evaluation Metrics ===");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:P2}");
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:P2}");
Console.WriteLine($"Log Loss: {metrics.LogLoss:F4}");
Console.WriteLine(
    $"Log Loss Reduction: {metrics.LogLossReduction:F4}");

// STEP 7: Create a prediction engine.
var predictionEngine =mlContext.Model.CreatePredictionEngine<
        TicketData, TicketPrediction>(model);

// STEP 8: Predict a new ticket.
var newTicket = new TicketData
{
    TicketText ="Payment was deducted twice and the invoice is incorrect"
};

var prediction = predictionEngine.Predict(newTicket);

Console.WriteLine("\n=== New Ticket Prediction ===");
Console.WriteLine($"Ticket: {newTicket.TicketText}");
Console.WriteLine($"Predicted category: {prediction.PredictedLabel}");

Console.WriteLine("\nRaw class scores:");
for (int i = 0; i < prediction.Score.Length; i++)
{
    Console.WriteLine($"Score {i}: {prediction.Score[i]:F4}");
}


// STEP 9: Save the trained pipeline.
string modelPath = Path.Combine(AppContext.BaseDirectory,"HelpDeskLightGbmModel.zip");

mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);

Console.WriteLine($"\nModel saved to: {modelPath}");

