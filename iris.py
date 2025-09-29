from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator

# Создаем Spark сессию
spark = SparkSession.builder \
    .appName("IrisAnalysis") \
    .getOrCreate()

IRIS_PATH = "/home/denis/.cache/kagglehub/datasets/uciml/iris/versions/2/Iris.csv"

# Загружаем данные с правильными именами столбцов
df = spark.read.csv(IRIS_PATH, header=True, inferSchema=True)

# Покажем структуру данных
print("Структура данных:")
df.printSchema()
print("Первые 10 строк:")
df.show(10)

# Предобработка данных
assembler = VectorAssembler(
    inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    outputCol="features"
)
df = assembler.transform(df)

# Индексация целевой переменной для классификации
label_indexer = StringIndexer(inputCol="Species", outputCol="label")
df = label_indexer.fit(df).transform(df)

# Удаляем ненужный столбец Id
df = df.drop("Id")

print("Данные после предобработки:")
df.select("features", "Species", "label").show(10, truncate=False)

# Разделение на тренировочную и тестовую выборки
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

print(f"Размер тренировочной выборки: {train_data.count()}")
print(f"Размер тестовой выборки: {test_data.count()}")

# КЛАССИФИКАЦИЯ
print("\n=== КЛАССИФИКАЦИЯ ===")

# Построение модели классификации (логистическая регрессия)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)

# Предсказание на тестовых данных
lr_predictions = lr_model.transform(test_data)

print("Результаты классификации:")
lr_predictions.select("Species", "label", "prediction", "probability").show()

# Оценка качества классификации
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "f1"})

print(f"Точность (Accuracy): {accuracy:.4f}")
print(f"F1-мера: {f1:.4f}")

# КЛАСТЕРИЗАЦИЯ
print("\n=== КЛАСТЕРИЗАЦИЯ ===")

# Кластеризация с помощью K-means
kmeans = KMeans(featuresCol="features", k=3, seed=42, maxIter=10)
kmeans_model = kmeans.fit(df)

# Предсказание кластеров
kmeans_predictions = kmeans_model.transform(df)

print("Результаты кластеризации:")
kmeans_predictions.select("Species", "features", "prediction").show(10)

# Оценка качества кластеризации
silhouette_evaluator = ClusteringEvaluator()
silhouette = silhouette_evaluator.evaluate(kmeans_predictions)

print(f"Оценка силуэта: {silhouette:.4f}")

# Показываем центры кластеров
centers = kmeans_model.clusterCenters()
print("Центры кластеров:")
for i, center in enumerate(centers):
    print(f"Кластер {i}: {center}")

# Анализ соответствия кластеров и видов ирисов
print("\nСоответствие кластеров и видов:")
kmeans_predictions.groupBy("Species", "prediction").count().orderBy("prediction", "Species").show()

# Дополнительная статистика по кластерам
print("\nСтатистика по кластерам:")
kmeans_predictions.groupBy("prediction").count().show()

# Останавливаем Spark сессию
spark.stop()
