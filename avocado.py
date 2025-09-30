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

path = "/home/denis/.cache/kagglehub/datasets/neuromusic/avocado-prices/versions/1/avocado.csv"

# Загрузка данных
df = spark.read.csv(path, header=True, inferSchema=True)

# Предобработка данных
df = df.drop("Index", "Date")  # Удаляем неинформативные столбцы

# Обработка категориальных признаков
indexer = StringIndexer(inputCol="type", outputCol="type_indexed")
encoder = OneHotEncoder(inputCol="type_indexed", outputCol="type_encoded")
region_indexer = StringIndexer(inputCol="region", outputCol="region_indexed")

# Подготовка фич для разных задач
numeric_cols = ["Total Volume", "4046", "4225", "4770", 
                "Total Bags", "Small Bags", "Large Bags", "XLarge Bags"]
all_features = numeric_cols + ["type_encoded", "region_indexed", "year"]

# Создание пайплайна предобработки
preprocessing_pipeline = Pipeline(stages=[
    indexer,
    encoder,
    region_indexer,
    VectorAssembler(inputCols=all_features, outputCol="features_all"),
    VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
])

# Выполнение предобработки
model_preproc = preprocessing_pipeline.fit(df)
df_processed = model_preproc.transform(df)

# Разделение на тренировочную и тестовую выборки
train_data, test_data = df_processed.randomSplit([0.7, 0.3], seed=42)

# 1. КЛАССИФИКАЦИЯ (предсказание типа авокадо)
print("\n=== КЛАССИФИКАЦИЯ ===")
lr = LogisticRegression(featuresCol="features_all", labelCol="type_indexed")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

# Метрики классификации
evaluator_multi = MulticlassClassificationEvaluator(labelCol="type_indexed", predictionCol="prediction")
print(f"Accuracy: {evaluator_multi.evaluate(lr_predictions, {evaluator_multi.metricName: 'accuracy'})}")
print(f"F1-score: {evaluator_multi.evaluate(lr_predictions, {evaluator_multi.metricName: 'f1'})}")

# 2. РЕГРЕССИЯ (предсказание средней цены)
print("\n=== РЕГРЕССИЯ ===")
regressor = LinearRegression(featuresCol="features_all", labelCol="AveragePrice")
lr_reg_model = regressor.fit(train_data)
reg_predictions = lr_reg_model.transform(test_data)

# Метрики регрессии
evaluator_reg = RegressionEvaluator(labelCol="AveragePrice", predictionCol="prediction")
print(f"RMSE: {evaluator_reg.evaluate(reg_predictions, {evaluator_reg.metricName: 'rmse'})}")
print(f"R2: {evaluator_reg.evaluate(reg_predictions, {evaluator_reg.metricName: 'r2'})}")

# 3. КЛАСТЕРИЗАЦИЯ (группировка по числовым признакам)
print("\n=== КЛАСТЕРИЗАЦИЯ ===")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features")
kmeans = KMeans(featuresCol="scaled_features", k=3, seed=42)

# Пайплайн для кластеризации
cluster_pipeline = Pipeline(stages=[scaler, kmeans])
kmeans_model = cluster_pipeline.fit(df_processed)
cluster_predictions = kmeans_model.transform(df_processed)

# Метрики кластеризации
evaluator_silhouette = ClusteringEvaluator(featuresCol="scaled_features", predictionCol="prediction")
silhouette_score = evaluator_silhouette.evaluate(cluster_predictions)
print(f"Silhouette Score: {silhouette_score}")

# Дополнительная информация о кластерах
centers = kmeans_model.stages[-1].clusterCenters()
print("Центры кластеров:")
for i, center in enumerate(centers):
    print(f"Кластер {i}: {center}")

spark.stop()
