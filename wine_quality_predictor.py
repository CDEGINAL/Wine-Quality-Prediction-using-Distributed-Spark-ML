# wine_quality_predictor.py

import logging
from pathlib import Path
from typing import List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    def __init__(self, model_dir: str = "/home/ec2-user/models"):
        """Initialize the Wine Quality Predictor.

        Args:
            model_dir: Directory for saving model and results
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "wine_model"
        self.results_path = self.model_dir / "results.txt"
        
    def create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        return SparkSession.builder \
            .appName("WineQualityPredictor") \
            .master("local[*]") \
            .getOrCreate()

    def load_data(self, spark: SparkSession, train_path: str, test_path: str) -> Tuple[DataFrame, DataFrame]:
        """Load and prepare training and test datasets.

        Args:
            spark: Active SparkSession
            train_path: Path to training dataset
            test_path: Path to test dataset

        Returns:
            Tuple of (training_dataframe, test_dataframe)
        """
        logger.info("Loading datasets...")
        data_train = spark.read.option("delimiter", ";") \
            .csv(train_path, header=True, inferSchema=True)
        data_test = spark.read.option("delimiter", ";") \
            .csv(test_path, header=True, inferSchema=True)
            
        return self._clean_data(data_train), self._clean_data(data_test)

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """Clean column names and filter unwanted values.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Clean column names
        for old_name in df.schema.names:
            clean_name = old_name.replace('"', '')
            if clean_name != old_name:
                df = df.withColumnRenamed(old_name, clean_name)
        
        # Filter out quality = 3
        return df.filter(df["quality"] != 3)

    def _get_feature_columns(self, df: DataFrame) -> List[str]:
        """Get list of feature columns excluding the target variable.

        Args:
            df: Input DataFrame

        Returns:
            List of feature column names
        """
        return [col for col in df.columns if col != "quality"]

    def create_pipeline(self, feature_cols: List[str]) -> Pipeline:
        """Create ML pipeline with feature processing and model.

        Args:
            feature_cols: List of feature column names

        Returns:
            Configured Pipeline object
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature")
        scaler = StandardScaler(inputCol="feature", outputCol="scaled_feature")
        classifier = LogisticRegression(labelCol="quality", featuresCol="scaled_feature")
        
        return Pipeline(stages=[assembler, scaler, classifier])

    def evaluate_model(self, predictions: DataFrame) -> Tuple[float, float]:
        """Calculate F1 score and accuracy for predictions.

        Args:
            predictions: DataFrame with predictions

        Returns:
            Tuple of (f1_score, accuracy)
        """
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", 
            predictionCol="prediction"
        )
        
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        
        return f1_score, accuracy

    def save_results(self, train_metrics: Tuple[float, float], test_metrics: Tuple[float, float]):
        """Save evaluation metrics to file.

        Args:
            train_metrics: Tuple of training (f1_score, accuracy)
            test_metrics: Tuple of test (f1_score, accuracy)
        """
        train_f1, train_acc = train_metrics
        test_f1, test_acc = test_metrics
        
        with open(self.results_path, "w") as fp:
            fp.write(f"[Train] F1 score = {train_f1:.4f}\n")
            fp.write(f"[Train] Accuracy = {train_acc:.4f}\n")
            fp.write(f"[Test] F1 score = {test_f1:.4f}\n")
            fp.write(f"[Test] Accuracy = {test_acc:.4f}\n")
        
        logger.info(f"Results saved to {self.results_path}")

    def train_and_evaluate(self, train_path: str, test_path: str):
        """Main method to train model and evaluate performance.

        Args:
            train_path: Path to training dataset
            test_path: Path to test dataset
        """
        try:
            spark = self.create_spark_session()
            
            # Load and prepare data
            data_train, data_test = self.load_data(spark, train_path, test_path)
            feature_cols = self._get_feature_columns(data_train)
            
            # Create and train pipeline
            logger.info("Training model...")
            pipeline = self.create_pipeline(feature_cols)
            model = pipeline.fit(data_train)
            
            # Save model
            logger.info(f"Saving model to {self.model_path}")
            model.write().overwrite().save(str(self.model_path))
            
            # Generate predictions
            train_predictions = model.transform(data_train)
            test_predictions = model.transform(data_test)
            
            # Evaluate model
            logger.info("Evaluating model performance...")
            train_metrics = self.evaluate_model(train_predictions)
            test_metrics = self.evaluate_model(test_predictions)
            
            # Save and display results
            self.save_results(train_metrics, test_metrics)
            logger.info(f"[Test] F1 score = {test_metrics[0]:.4f}")
            logger.info(f"[Test] Accuracy = {test_metrics[1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}", exc_info=True)
            raise
        
        finally:
            spark.stop()

def main():
    """Main entry point of the application."""
    predictor = WineQualityPredictor()
    predictor.train_and_evaluate(
        train_path="TrainingDataset.csv",
        test_path="ValidationDataset.csv"
    )

if __name__ == "__main__":
    main()
