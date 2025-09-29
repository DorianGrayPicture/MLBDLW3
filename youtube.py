from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

# Инициализация SparkSession
spark = SparkSession.builder \
    .appName("YouTube Channels Analysis") \
    .getOrCreate()

# Загрузка данных
df = spark.read.csv("/home/denis/.cache/kagglehub/datasets/asaniczka/2024-youtube-channels-1-million/versions/1/youtube_channels_1M_clean.csv", header=True, inferSchema=True)

# Показать схему данных для отладки
print("Исходная схема данных:")
df.printSchema()

# Предварительная обработка данных с преобразованием типов
cleaned_df = df \
    .drop("channel_id", "channel_link", "channel_name", "banner_link", "avatar", "description") \
    .filter(col("subscriber_count").isNotNull()) \
    .filter(col("total_views").isNotNull()) \
    .filter(col("country").isNotNull()) \
    .withColumn("subscriber_count", col("subscriber_count").cast(DoubleType())) \
    .withColumn("total_views", 
                when(col("total_views").rlike("^[0-9.]+$"), col("total_views").cast(DoubleType()))
                .otherwise(None)) \
    .filter(col("subscriber_count") > 0) \
    .filter(col("total_views") > 0) \
    .withColumn("views_per_subscriber", col("total_views") / col("subscriber_count")) \
    .filter(col("views_per_subscriber").isNotNull()) \
    .filter(col("views_per_subscriber") > 0)

print("Данные после очистки:")
cleaned_df.show(5)
cleaned_df.printSchema()

# 1. РЕГРЕССИЯ: Предсказание subscriber_count на основе total_views
print("\n=== РЕГРЕССИЯ ===")
reg_data = cleaned_df.select("subscriber_count", "total_views")

assembler_reg = VectorAssembler(
    inputCols=["total_views"],
    outputCol="features_reg"
)

# Разделение на тренировочные и тестовые данные
train_reg, test_reg = reg_data.randomSplit([0.8, 0.2], seed=42)

# Создание и обучение модели регрессии
lr = LinearRegression(
    featuresCol="features_reg",
    labelCol="subscriber_count",
    predictionCol="predicted_subscribers",
    maxIter=10
)

pipeline_reg = Pipeline(stages=[assembler_reg, lr])
model_reg = pipeline_reg.fit(train_reg)

# Оценка модели регрессии
predictions_reg = model_reg.transform(test_reg)
print("Регрессия - Примеры предсказаний:")
predictions_reg.select("subscriber_count", "predicted_subscribers").show(5)

# Вывод метрик регрессии
training_summary = model_reg.stages[-1].summary
print(f"R²: {training_summary.r2}")
print(f"RMSE: {training_summary.rootMeanSquaredError}")

# 2. КЛАССИФИКАЦИЯ: Классификация по странам (топ-5 стран)
print("\n=== КЛАССИФИКАЦИЯ ===")
# Выбор топ-5 стран по количеству каналов
top_countries = cleaned_df \
    .groupBy("country") \
    .count() \
    .orderBy(desc("count")) \
    .limit(5) \
    .select("country") \
    .rdd.flatMap(lambda x: x).collect()

print(f"Топ-5 стран для классификации: {top_countries}")

classification_data = cleaned_df \
    .filter(col("country").isin(top_countries)) \
    .select("country", "subscriber_count", "total_views", "views_per_subscriber")

# Индексация целевой переменной
indexer = StringIndexer(
    inputCol="country",
    outputCol="label",
    handleInvalid="skip"
)

assembler_clf = VectorAssembler(
    inputCols=["subscriber_count", "total_views", "views_per_subscriber"],
    outputCol="features_clf"
)

# Разделение данных
train_clf, test_clf = classification_data.randomSplit([0.8, 0.2], seed=42)

# Создание и обучение модели классификации
lr_clf = LogisticRegression(
    featuresCol="features_clf",
    labelCol="label",
    predictionCol="predicted_country",
    maxIter=10
)

pipeline_clf = Pipeline(stages=[indexer, assembler_clf, lr_clf])
model_clf = pipeline_clf.fit(train_clf)

# Оценка модели классификации
predictions_clf = model_clf.transform(test_clf)
print("Классификация - Точность:",
      model_clf.stages[-1].summary.accuracy)

# Показать распределение по классам
print("Распределение по странам:")
predictions_clf.groupBy("country", "predicted_country").count().show()

# 3. КЛАСТЕРИЗАЦИЯ: Группировка каналов по характеристикам
print("\n=== КЛАСТЕРИЗАЦИЯ ===")
cluster_data = cleaned_df \
    .select("subscriber_count", "total_views", "views_per_subscriber") \
    .filter((col("subscriber_count") > 1000) & 
            (col("subscriber_count") < 1e8) &
            (col("total_views") < 1e10))  # Фильтр для исключения аномалий

# Масштабирование признаков
assembler_cluster = VectorAssembler(
    inputCols=["subscriber_count", "total_views", "views_per_subscriber"],
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_cluster",
    withStd=True,
    withMean=True
)

# Создание и обучение K-means модели
kmeans = KMeans(
    featuresCol="features_cluster",
    predictionCol="cluster",
    k=4,  # Уменьшили количество кластеров для стабильности
    seed=42,
    maxIter=20
)

pipeline_cluster = Pipeline(stages=[assembler_cluster, scaler, kmeans])
model_cluster = pipeline_cluster.fit(cluster_data)

# Получение результатов кластеризации
predictions_cluster = model_cluster.transform(cluster_data)
print("Кластеризация - Размеры кластеров:")
predictions_cluster.groupBy("cluster").count().orderBy("cluster").show()

# Статистика по кластерам
print("Статистика по кластерам:")
predictions_cluster.groupBy("cluster").agg(
    mean("subscriber_count").alias("avg_subscribers"),
    mean("total_views").alias("avg_views"),
    mean("views_per_subscriber").alias("avg_views_per_sub")
).orderBy("cluster").show()

# Вычисление центров кластеров
centers = model_cluster.stages[-1].clusterCenters()
print("Центры кластеров (масштабированные):")
for i, center in enumerate(centers):
    print(f"Кластер {i}: {center}")

# Дополнительный анализ: основные статистики
print("\n=== ОСНОВНЫЕ СТАТИСТИКИ ДАТАСЕТА ===")
cleaned_df.select("subscriber_count", "total_views", "views_per_subscriber").describe().show()

# Анализ по странам
print("Топ-10 стран по количеству каналов:")
cleaned_df.groupBy("country").count().orderBy(desc("count")).show(10)

# Остановка SparkSession
spark.stop()
