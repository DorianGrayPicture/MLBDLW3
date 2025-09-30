from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import *
from pyspark.sql.functions import *

# Инициализация Spark
spark = SparkSession.builder \
    .appName("AvocadoAnalysis") \
    .getOrCreate()

# Загрузка данных
df = spark.read.csv("/home/denis/.cache/kagglehub/datasets/neuromusic/avocado-prices/versions/1/avocado.csv", header=True, inferSchema=True)

# Предобработка данных
df = df.drop("Index", "Date")  # Удаляем неинформативные столбцы

# Обработка категориальных признаков
type_indexer = StringIndexer(inputCol="type", outputCol="type_indexed")
region_indexer = StringIndexer(inputCol="region", outputCol="region_indexed")

# Подготовка фич для разных задач
numeric_cols = ["Total Volume", "4046", "4225", "4770", 
                "Total Bags", "Small Bags", "Large Bags", "XLarge Bags"]

# Разделение данных ДО предобработки чтобы избежать утечки
train_data_raw, test_data_raw = df.randomSplit([0.7, 0.3], seed=42)

# 1. КЛАССИФИКАЦИЯ (предсказание типа авокадо)
print("\n=== КЛАССИФИКАЦИЯ ===")

# Для классификации используем только числовые признаки + год, исключаем регион чтобы избежать утечки
classification_features = numeric_cols + ["year"]

# Создаем пайплайн для классификации
classification_pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=classification_features, outputCol="features_clf"),
    LogisticRegression(featuresCol="features_clf", labelCol="type_indexed")
])

# Обучаем на тренировочных данных
type_indexer_model = type_indexer.fit(train_data_raw)
train_data_clf = type_indexer_model.transform(train_data_raw)
test_data_clf = type_indexer_model.transform(test_data_raw)

lr_model = classification_pipeline.fit(train_data_clf)
lr_predictions = lr_model.transform(test_data_clf)

# Метрики классификации
evaluator_multi = MulticlassClassificationEvaluator(labelCol="type_indexed", predictionCol="prediction")
print(f"Accuracy: {evaluator_multi.evaluate(lr_predictions, {evaluator_multi.metricName: 'accuracy'})}")
print(f"F1-score: {evaluator_multi.evaluate(lr_predictions, {evaluator_multi.metricName: 'f1'})}")

# Покажем распределение предсказаний
print("Распределение предсказаний:")
lr_predictions.groupBy("prediction", "type_indexed").count().show()

# 2. РЕГРЕССИЯ (предсказание средней цены)
print("\n=== РЕГРЕССИЯ ===")

# Для регрессии используем все признаки кроме цены
regression_features = numeric_cols + ["year"]

regression_pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=regression_features, outputCol="features_reg"),
    LinearRegression(featuresCol="features_reg", labelCol="AveragePrice")
])

lr_reg_model = regression_pipeline.fit(train_data_raw)
reg_predictions = lr_reg_model.transform(test_data_raw)

# Метрики регрессии
evaluator_reg = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction")
print(f"RMSE: {evaluator_reg.evaluate(reg_predictions, {evaluator_reg.metricName: 'rmse'})}")
print(f"R2: {evaluator_reg.evaluate(reg_predictions, {evaluator_reg.metricName: 'r2'})}")
print(f"MAE: {evaluator_reg.evaluate(reg_predictions, {evaluator_reg.metricName: 'mae'})}")

# 3. КЛАСТЕРИЗАЦИЯ (группировка по числовым признакам)
print("\n=== КЛАСТЕРИЗАЦИЯ ===")

# Для кластеризации используем только основные числовые признаки
clustering_features = ["Total Volume", "4046", "4225", "4770", "AveragePrice"]

cluster_pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=clustering_features, outputCol="features_cluster"),
    StandardScaler(inputCol="features_cluster", outputCol="scaled_features"),
    KMeans(featuresCol="scaled_features", k=3, seed=42)
])

kmeans_model = cluster_pipeline.fit(df)
cluster_predictions = kmeans_model.transform(df)

# Метрики кластеризации
evaluator_silhouette = ClusteringEvaluator(featuresCol="scaled_features", predictionCol="prediction")
silhouette_score = evaluator_silhouette.evaluate(cluster_predictions)
print(f"Silhouette Score: {silhouette_score}")

# Анализ кластеров
print("Размеры кластеров:")
cluster_predictions.groupBy("prediction").count().orderBy("prediction").show()

# Статистика по кластерам
print("Средние значения по кластерам:")
cluster_stats = cluster_predictions.groupBy("prediction").agg(
    mean("AveragePrice").alias("avg_price"),
    mean("Total Volume").alias("avg_volume"),
    mean("4046").alias("avg_4046"),
    mean("4225").alias("avg_4225"),
    mean("4770").alias("avg_4770")
)
cluster_stats.show()

# Визуализация связи кластеров с типом авокадо
print("Распределение типов по кластерам:")
cluster_predictions.groupBy("prediction", "type").count().orderBy("prediction", "type").show()

spark.stop()
