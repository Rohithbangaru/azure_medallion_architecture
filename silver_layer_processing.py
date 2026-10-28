# Databricks notebook source
# CELL 1: Storage Authentication & Path Setup
from pyspark.sql.functions import col, to_timestamp, current_timestamp, row_number
from pyspark.sql.window import Window
from delta.tables import DeltaTable

config={}
with open(".env", "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        key, value = line.strip().split("=", 1)
        config[key] = value
STORAGE_ACCOUNT = config.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_KEY = config.get("AZURE_STORAGE_ACCOUNT_KEY")

# Configure storage credentials
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    STORAGE_KEY
)

# Paths
BRONZE_DELTA_PATH  = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/delta/orders_bronze"
SILVER_BASE_PATH   = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_ORDERS_PATH = f"{SILVER_BASE_PATH}/delta/orders_silver"
QUARANTINE_PATH    = f"{SILVER_BASE_PATH}/quarantine/orders_bad_records"

# COMMAND ----------

# CELL 2: Ingest from Bronze Layer
# Read only the latest changes or batch from Bronze Delta
bronze_df = spark.read.format("delta").load(BRONZE_DELTA_PATH)
print(f"Total Bronze Records Ingested: {bronze_df.count()}")

# COMMAND ----------

# CELL 3: Type Casting & Cleaning
cleaned_df = (
    bronze_df
    .withColumn("order_id", col("order_id").cast("int"))
    .withColumn("amount", col("amount").cast("double"))
    .withColumn("order_timestamp", to_timestamp(col("order_date"), "yyyy-MM-dd HH:mm:ss"))
    .drop("order_date")
)

# COMMAND ----------

# CELL 4: Data Quality Check & Quarantine Split
# Rule: order_id, customer_id, and amount must NOT be NULL, and amount must be > 0
valid_condition = (
    col("order_id").isNotNull() & 
    col("customer_id").isNotNull() & 
    col("amount").isNotNull() & 
    (col("amount") > 0)
)

valid_df = cleaned_df.filter(valid_condition)
quarantine_df = (
    cleaned_df
    .filter(~valid_condition)
    .withColumn("_quarantined_at", current_timestamp())
    .withColumn("_rejection_reason", col("amount").isNull().cast("string"))
)

# Write bad/quarantined data for downstream auditing
(
    quarantine_df.write
    .format("delta")
    .mode("append")
    .save(QUARANTINE_PATH)
)

print(f"Quarantined Records: {quarantine_df.count()} (Saved to {QUARANTINE_PATH})")

# COMMAND ----------

# CELL 5: Deduplication using Window Functions
# Handle duplicate rows (like duplicate order_id 1005 in our sample)
window_spec = Window.partitionBy("order_id").orderBy(col("_ingested_at").desc())

deduped_df = (
    valid_df
    .withColumn("row_num", row_number().over(window_spec))
    .filter(col("row_num") == 1)
    .drop("row_num")
    .withColumn("_processed_at", current_timestamp())
)

print(f"Valid Deduplicated Records: {deduped_df.count()}")
display(deduped_df)

# COMMAND ----------

# CELL 6: Delta Lake MERGE (UPSERT)
# If target Silver table does not exist, initialize it; otherwise execute MERGE
if not DeltaTable.isDeltaTable(spark, SILVER_ORDERS_PATH):
    print("Silver table does not exist. Initializing table...")
    (
        deduped_df.write
        .format("delta")
        .mode("overwrite")
        .save(SILVER_ORDERS_PATH)
    )
else:
    print("Silver table exists. Executing MERGE...")
    silver_table = DeltaTable.forPath(spark, SILVER_ORDERS_PATH)
    
    (
        silver_table.alias("target")
        .merge(
            deduped_df.alias("source"),
            "target.order_id = source.order_id"
        )
        .whenMatchedUpdate(set={
            "customer_id": "source.customer_id",
            "product_id": "source.product_id",
            "amount": "source.amount",
            "status": "source.status",
            "order_timestamp": "source.order_timestamp",
            "_processed_at": "source._processed_at"
        })
        .whenNotMatchedInsertAll()
        .execute()
    )

print("MERGE execution completed successfully.")

# COMMAND ----------

# CELL 7: Verification & Delta History Check
silver_result = spark.read.format("delta").load(SILVER_ORDERS_PATH)
display(silver_result)

# Time travel history verification to confirm MERGE or WRITE operation
display(spark.sql(f"DESCRIBE HISTORY delta.`{SILVER_ORDERS_PATH}`"))