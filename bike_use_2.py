from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np

# Инициализация Spark сессии
spark = SparkSession.builder \
    .appName("MiBici Analysis") \
    .getOrCreate()

# Схема данных
schema = StructType([
    StructField("Index", IntegerType()),
    StructField("Trip_Id", IntegerType()),
    StructField("User_Id", IntegerType()),
    StructField("Sex", StringType()),
    StructField("Birth_year", IntegerType()),
    StructField("Trip_start", TimestampType()),
    StructField("Trip_end", TimestampType()),
    StructField("Origin_Id", IntegerType()),
    StructField("Destination_Id", IntegerType()),
    StructField("Age", IntegerType()),
    StructField("Duration", StringType())
])

bike_use_path = "/home/denis/.cache/kagglehub/datasets/sebastianquirarte/over-9-years-of-real-public-bike-use-data-mibici/versions/3/mibici_2014-2024/mibici_2014-2024.csv"
# Чтение данных
df = spark.read.csv(bike_use_path, header=True, schema=schema)

# Преобразование длительности в секунды
df = df.withColumn("Duration_sec", 
    expr("coalesce(cast(substring(Duration, 10, 2) as int)*3600 + " +
         "cast(substring(Duration, 13, 2) as int)*60 + " +
         "cast(substring(Duration, 16, 2) as int), 0)"))

# Создание временных признаков
df = df.withColumn("hour", hour("Trip_start")) \
       .withColumn("day_of_week", dayofweek("Trip_start")) \
       .withColumn("month", month("Trip_start"))

# Фильтрация некорректных данных
df_clean = df.filter(
    (col("Age") >= 16) & (col("Age") <= 80) &
    (col("Duration_sec") > 0) & (col("Duration_sec") <= 3600) &  # до 1 часа
    (col("Sex").isin(["M", "F"]))
).cache()

print("Базовая информация о данных:")
df_clean.select("Age", "Duration_sec", "Sex", "hour").describe().show()

# 1. КЛАССИФИКАЦИЯ (предсказание пола по возрасту и времени поездки)
print("\n=== КЛАССИФИКАЦИЯ ===")

# Подготовка данных
indexer = StringIndexer(inputCol="Sex", outputCol="label")
assembler = VectorAssembler(
    inputCols=["Age", "Duration_sec", "hour", "day_of_week"],
    outputCol="features"
)

lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[indexer, assembler, lr])

# Разделение на тренировочные и тестовые данные
train_data, test_data = df_clean.randomSplit([0.7, 0.3], seed=42)

# Обучение модели
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# Оценка точности
accuracy = predictions.filter(col("label") == col("prediction")).count() / test_data.count()
print(f"Точность классификации: {accuracy:.2%}")

# 2. КЛАСТЕРИЗАЦИЯ (группировка поездок по паттернам использования)
print("\n=== КЛАСТЕРИЗАЦИЯ ===")

kmeans = KMeans(
    featuresCol="features",
    k=5,  # 5 кластеров
    seed=42
)

# Используем те же признаки, но без целевой переменной
cluster_data = assembler.transform(df_clean.limit(100000))  # Ограничиваем для скорости

kmeans_model = kmeans.fit(cluster_data)
clustered = kmeans_model.transform(cluster_data)

print("Размеры кластеров:")
clustered.groupBy("prediction").count().orderBy("prediction").show()

# Центроиды кластеров
print("Центроиды кластеров:")
centers = kmeans_model.clusterCenters()
for i, center in enumerate(centers):
    print(f"Кластер {i}: {center}")

# 3. РЕГРЕССИЯ (предсказание длительности поездки)
print("\n=== РЕГРЕССИЯ ===")

# Подготовка данных для регрессии
reg_assembler = VectorAssembler(
    inputCols=["Age", "hour", "day_of_week", "month"],
    outputCol="reg_features"
)

lr_reg = LinearRegression(
    featuresCol="reg_features",
    labelCol="Duration_sec",
    maxIter=10
)

pipeline_reg = Pipeline(stages=[reg_assembler, lr_reg])

# Разделение данных
train_reg, test_reg = df_clean.randomSplit([0.7, 0.3], seed=42)

# Обучение модели
reg_model = pipeline_reg.fit(train_reg)
reg_pred = reg_model.transform(test_reg)

# Создание evaluator для различных метрик
evaluator_rmse = RegressionEvaluator(
    labelCol="Duration_sec", 
    predictionCol="prediction", 
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="Duration_sec", 
    predictionCol="prediction", 
    metricName="mae"
)

evaluator_mse = RegressionEvaluator(
    labelCol="Duration_sec", 
    predictionCol="prediction", 
    metricName="mse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="Duration_sec", 
    predictionCol="prediction", 
    metricName="r2"
)

# Вычисление всех метрик
rmse = evaluator_rmse.evaluate(reg_pred)
mae = evaluator_mae.evaluate(reg_pred)
mse = evaluator_mse.evaluate(reg_pred)
r2 = evaluator_r2.evaluate(reg_pred)

# Дополнительные метрики через pandas для простоты
reg_pred_pd = reg_pred.select("Duration_sec", "prediction").toPandas()
actual = reg_pred_pd["Duration_sec"]
predicted = reg_pred_pd["prediction"]

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Вывод всех метрик регрессии
print("\n=== МЕТРИКИ РЕГРЕССИИ ===")
print(f"R² (Коэффициент детерминации): {r2:.4f}")
print(f"MSE (Средняя квадратичная ошибка): {mse:.2f}")
print(f"RMSE (Среднеквадратичная ошибка): {rmse:.2f} секунд")
print(f"MAE (Средняя абсолютная ошибка): {mae:.2f} секунд")
print(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")

# Статистика по целевой переменной для контекста
stats = df_clean.select(
    mean("Duration_sec").alias("mean_duration"),
    stddev("Duration_sec").alias("std_duration"),
    min("Duration_sec").alias("min_duration"),
    max("Duration_sec").alias("max_duration")
).collect()

mean_dur = stats[0]["mean_duration"]
std_dur = stats[0]["std_duration"]

print(f"\nСтатистика длительности поездок:")
print(f"Средняя длительность: {mean_dur:.2f} секунд")
print(f"Стандартное отклонение: {std_dur:.2f} секунд")
print(f"RMSE/StdDev ratio: {rmse/std_dur:.3f}")

# Анализ важности признаков
lr_model = reg_model.stages[-1]
print(f"\nКоэффициенты модели:")
features = ["Age", "Hour", "Day_of_week", "Month"]
for feature, coef in zip(features, lr_model.coefficients):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {lr_model.intercept:.4f}")

# Вывод примеров предсказаний
print("\nПримеры предсказаний регрессии (первые 10):")
reg_pred.select("Age", "hour", "Duration_sec", "prediction", 
               abs(col("Duration_sec") - col("prediction")).alias("error")) \
        .orderBy(rand()) \
        .show(10)

# Анализ ошибок по группам
print("Средняя ошибка по возрасту:")
reg_pred.groupBy(
    when(col("Age") < 25, "18-24")
    .when(col("Age") < 35, "25-34") 
    .when(col("Age") < 45, "35-44")
    .when(col("Age") < 55, "45-54")
    .otherwise("55+").alias("age_group")
).agg(
    mean(abs(col("Duration_sec") - col("prediction"))).alias("avg_abs_error"),
    mean("Duration_sec").alias("avg_actual_duration"),
    count("*").alias("count")
).orderBy("age_group").show()

spark.stop()
