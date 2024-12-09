from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os
from typing import Tuple, List

def create_spark_session() -> SparkSession:
    """Create and return a Spark session"""
    return SparkSession.builder.master("local[*]").getOrCreate()

def load_data(spark: SparkSession, train_path: str, test_path: str = None) -> Tuple[DataFrame, DataFrame]:
    """Load training and test datasets"""
    try:
        data_train = spark.read.option("delimiter", ";").csv(train_path, header=True, inferSchema=True)
        
        if test_path:
            data_test = spark.read.option("delimiter", ";").csv(test_path, header=True, inferSchema=True)
        else:
            data_test = spark.read.option("delimiter", ";").csv('ValidationDataset.csv', header=True, inferSchema=True)
        
        return data_train, data_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def clean_data(data_train: DataFrame, data_test: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Clean column names and filter training data"""
    # Clean column names
    for old_col in data_train.schema.names:
        clean_col = old_col.replace('"', '')
        if clean_col != old_col:
            data_train = data_train.withColumnRenamed(old_col, clean_col)
            data_test = data_test.withColumnRenamed(old_col, clean_col)

    # Filter out quality = 3
    data_train = data_train.filter(data_train['quality'] != "3")
    
    return data_train, data_test

def create_pipeline(feature_cols: List[str]) -> Pipeline:
    """Create ML pipeline with feature processing and model"""
    return Pipeline(stages=[
        VectorAssembler(inputCols=feature_cols, outputCol="feature"),
        StandardScaler(inputCol="feature", outputCol="Scaled_feature"),
        LogisticRegression(labelCol="quality", featuresCol="Scaled_feature")
    ])

def evaluate_model(predictions: DataFrame) -> Tuple[float, float]:
    """Calculate F1 score and accuracy"""
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction"
    )
    
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    
    return f1_score, accuracy

def save_results(train_metrics: Tuple[float, float], test_metrics: Tuple[float, float], 
                save_path: str = "/job/results.txt"):
    """Save evaluation metrics to file"""
    train_f1, train_acc = train_metrics
    test_f1, test_acc = test_metrics
    
    with open(save_path, "w") as f:
        f.write("="*70 + "\n")
        f.write(f"[Train] F1 Score: {train_f1:.4f}\n")
        f.write(f"[Train] Accuracy: {train_acc:.4f}\n")
        f.write("="*70 + "\n")
        f.write(f"[Test] F1 Score: {test_f1:.4f}\n")
        f.write(f"[Test] Accuracy: {test_acc:.4f}\n")
        f.write("="*70 + "\n")

def main():
    """Main function to run the training pipeline"""
    spark = create_spark_session()
    
    try:
        # Load data
        test_path = f"/job/{sys.argv[1]}" if len(sys.argv) == 2 else None
        data_train, data_test = load_data(spark, 'TrainingDataset.csv', test_path)
        
        # Clean data
        data_train, data_test = clean_data(data_train, data_test)
        
        # Prepare feature columns
        feature_cols = [x for x in data_train.columns if x != "quality"]
        
        # Create and train pipeline
        pipeline = create_pipeline(feature_cols)
        model = pipeline.fit(data_train)
        
        # Save model
        model.write().overwrite().save("/job/Modelfile")
        
        # Generate predictions
        train_predictions = model.transform(data_train)
        test_predictions = model.transform(data_test)
        
        # Evaluate model
        train_metrics = evaluate_model(train_predictions)
        test_metrics = evaluate_model(test_predictions)
        
        # Save and display results
        save_results(train_metrics, test_metrics)
        
        print("="*70)
        print(f"[Train] F1 Score: {train_metrics[0]:.4f}")
        print(f"[Train] Accuracy: {train_metrics[1]:.4f}")
        print("="*70)
        print(f"[Test] F1 Score: {test_metrics[0]:.4f}")
        print(f"[Test] Accuracy: {test_metrics[1]:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
