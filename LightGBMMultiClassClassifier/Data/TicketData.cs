using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace LightGBMMultiClassClassifier.Data
{
    public class TicketData
    {
        [LoadColumn(0)]
        public string TicketText { get; set; } = "";
        [LoadColumn(1)]
        public string Category { get; set; } = "";
    }
}
