# Databricks notebook source
# CELL 1: Configuration & Storage Authentication
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    IntegerType, DoubleType, TimestampType
)
from pyspark.sql.functions import current_timestamp, input_file_name


config={}
with open(".env", "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        key, value = line.strip().split("=", 1)
        config[key] = value
STORAGE_ACCOUNT = config.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_KEY = config.get("AZURE_STORAGE_ACCOUNT_KEY")
# Configure ADLS Gen2 connectivity
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    STORAGE_KEY
)

# Define Base Paths
BRONZE_BASE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
RAW_DATA_PATH    = f"{BRONZE_BASE_PATH}/raw_orders"
BRONZE_DELTA_PATH = f"{BRONZE_BASE_PATH}/delta/orders_bronze"
CHECKPOINT_PATH  = f"{BRONZE_BASE_PATH}/checkpoints/orders_bronze"

# COMMAND ----------

# CELL 2: Define Strict Raw Schema
# Defining schema explicitly prevents runtime schema-inference overhead and data corruption.
orders_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("order_date", StringType(), True), # Keep raw string to parse cleanly in Silver
    StructField("amount", DoubleType(), True),
    StructField("status", StringType(), True)
])

# COMMAND ----------

# CELL 3: Read Raw Ingestion with Audit Columns
raw_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .schema(orders_schema)
    .load(f"{RAW_DATA_PATH}/*.csv")
)

# Add lineage and operational metadata columns
bronze_df = (
    raw_df
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_file", input_file_name())
)

print("Sample records from raw batch ingestion:")
display(bronze_df)

# COMMAND ----------

# CELL 4: Write to Delta Lake (Append Mode)
# Write the data in Delta format to the bronze storage directory
(
    bronze_df.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "false") # Protects against unexpected raw schema mutations
    .save(BRONZE_DELTA_PATH)
)

print(f"Bronze Delta ingestion complete. Target: {BRONZE_DELTA_PATH}")

# COMMAND ----------

# CELL 5: Verify Delta Lake Ingestion & History (Works seamlessly with Single User / UC)

# 1. Query the Delta files directly using Spark SQL
display(spark.sql(f"SELECT * FROM delta.`{BRONZE_DELTA_PATH}`"))

# 2. Register as a temporary view for straightforward SQL queries
bronze_df = spark.read.format("delta").load(BRONZE_DELTA_PATH)
bronze_df.createOrReplaceTempView("bronze_orders")

# Query the view
spark.sql("SELECT * FROM bronze_orders").show(truncate=False)

# 3. Verify Delta Transaction Log and Time Travel metadata
display(spark.sql(f"DESCRIBE HISTORY delta.`{BRONZE_DELTA_PATH}`"))