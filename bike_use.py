from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

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
       .withColumn("day_of_week", dayofweek("Trip_start"))

# Фильтрация некорректных данных
df_clean = df.filter(
    (col("Age") >= 16) & (col("Age") <= 80) &
    (col("Duration_sec") > 0) &
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

# 3. РЕГРЕССИЯ (предсказание длительности поездки)
print("\n=== РЕГРЕССИЯ ===")

# Подготовка данных для регрессии
reg_assembler = VectorAssembler(
    inputCols=["Age", "hour", "day_of_week"],
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

# Оценка модели
r2 = reg_model.stages[-1].summary.r2
print(f"R2 score регрессии: {r2:.3f}")

# Вывод примеров предсказаний
print("Примеры предсказаний регрессии:")
reg_pred.select("Age", "hour", "Duration_sec", "prediction").show(10)

spark.stop()
