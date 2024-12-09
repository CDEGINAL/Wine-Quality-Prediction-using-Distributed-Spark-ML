from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder.master("local[*]").getOrCreate()

def load_and_clean_data(spark, file_path):
    """Load and clean the test data"""
    try:
        data = spark.read.option("delimiter", ";").csv(file_path, header=True, inferSchema=True)
        
        # Clean column names
        for old_col in data.schema.names:
            clean_col = old_col.replace('"', '')
            if clean_col != old_col:
                data = data.withColumnRenamed(old_col, clean_col)
        
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please ensure the CSV file is properly formatted and accessible")
        sys.exit(1)

def load_model(model_path):
    """Load the trained model"""
    try:
        return PipelineModel.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure the model file is present in the specified path")
        sys.exit(1)

def evaluate_predictions(predictions):
    """Calculate and return evaluation metrics"""
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction"
    )
    
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    
    return f1_score, accuracy

def save_results(predictions, metrics, results_path="/job"):
    """Save predictions and metrics"""
    f1_score, accuracy = metrics
    
    # Save predictions
    predictions.drop("feature", "Scaled_feature", "rawPrediction", "probability") \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(os.path.join(results_path, "resultdata.csv"))
    
    # Save metrics
    with open(os.path.join(results_path, "results.txt"), "w") as f:
        f.write("="*70 + "\n")
        f.write(f"[Test] F1 Score: {f1_score:.4f}\n")
        f.write(f"[Test] Accuracy: {accuracy:.4f}\n")
        f.write("="*70 + "\n")

def main():
    """Main function to run the prediction pipeline"""
    if len(sys.argv) != 2:
        print("Usage: python wine_test.py <test_file.csv>")
        print("Example: docker run -v /path/to/data:/job wine-predictor TestDataset.csv")
        sys.exit(1)

    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load and process test data
        test_file = os.path.join("/job", sys.argv[1])
        test_data = load_and_clean_data(spark, test_file)
        
        # Load model and make predictions
        model = load_model("/job/Modelfile")
        predictions = model.transform(test_data)
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions)
        
        # Save results
        save_results(predictions, metrics)
        
        # Print metrics
        print("="*70)
        print(f"[Test] F1 Score: {metrics[0]:.4f}")
        print(f"[Test] Accuracy: {metrics[1]:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
