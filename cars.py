from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator

# Создаем Spark сессию
spark = SparkSession.builder \
    .appName("VehicleAnalysis") \
    .getOrCreate()

path = "/home/denis/.cache/kagglehub/datasets/jitikpatel/used-cars-price-dataset/versions/1/output_updated.csv"

# Загружаем данные
df = spark.read.csv(path, header=True, inferSchema=True)

# Предобработка данных
# Удаляем пропуски
df = df.na.drop()

# Индексация категориальных признаков
categorical_cols = ["Make", "Model", "Body Type", "Cylinders", "Transmission", "Fuel Type", "Color", "Location"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in categorical_cols]

# One-Hot Encoding для категориальных признаков
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_cols]

# Подготовка фичей для моделей
feature_cols = ["Year", "Mileage"] + [col+"_encoded" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Разделение данных
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 1. РЕГРЕССИЯ (предсказание Price)
rf_regressor = RandomForestRegressor(
    featuresCol="features",
    labelCol="Price",
    numTrees=10,
    seed=42
)

regression_pipeline = Pipeline(stages=indexers + encoders + [assembler, rf_regressor])
regression_model = regression_pipeline.fit(train_data)
regression_predictions = regression_model.transform(test_data)

# Метрики для регрессии
regression_evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction")
rmse = regression_evaluator.setMetricName("rmse").evaluate(regression_predictions)
r2 = regression_evaluator.setMetricName("r2").evaluate(regression_predictions)

print("Регрессия - Метрики:")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}\n")

# 2. КЛАССИФИКАЦИЯ (предсказание Condition)
condition_indexer = StringIndexer(inputCol="Condition", outputCol="label", handleInvalid="skip")
classifier_feature_cols = ["Year", "Mileage"] + [col+"_encoded" for col in categorical_cols]
classifier_assembler = VectorAssembler(inputCols=classifier_feature_cols, outputCol="classifier_features")

rf_classifier = RandomForestClassifier(
    featuresCol="classifier_features",
    labelCol="label",
    numTrees=10,
    seed=42
)

classification_pipeline = Pipeline(stages=indexers + encoders + [condition_indexer, classifier_assembler, rf_classifier])
classification_model = classification_pipeline.fit(train_data)
classification_predictions = classification_model.transform(test_data)

# Метрики для классификации
classification_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = classification_evaluator.setMetricName("accuracy").evaluate(classification_predictions)
f1 = classification_evaluator.setMetricName("f1").evaluate(classification_predictions)

print("Классификация - Метрики:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}\n")

# 3. КЛАСТЕРИЗАЦИЯ (группировка по характеристикам)
kmeans = KMeans(
    featuresCol="features",
    k=3,  # Количество кластеров
    seed=42
)

clustering_pipeline = Pipeline(stages=indexers + encoders + [assembler, kmeans])
clustering_model = clustering_pipeline.fit(train_data)
clustering_predictions = clustering_model.transform(test_data)

# Метрики для кластеризации
clustering_evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")
silhouette = clustering_evaluator.evaluate(clustering_predictions)

print("Кластеризация - Метрики:")
print(f"Silhouette Score: {silhouette:.4f}")

# Показываем примеры предсказаний
print("\nПримеры предсказаний:")
clustering_predictions.select("Make", "Model", "Year", "Price", "prediction").show(10)

# Останавливаем Spark сессию
spark.stop()
