using System;
using System.Collections.Generic;
using System.Text;

namespace LightGBMMultiClassClassifier.Data
{
    public class TicketPrediction
    {
        // Predicted category after mapping the key back to text.
        public string PredictedLabel { get; set; } = "";

        // One score per class, in the model's class order.
        public float[] Score { get; set; } = [];
    }

}
