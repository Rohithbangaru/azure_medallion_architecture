# Databricks notebook source
# CELL 1: Storage Authentication & Path Setup
from pyspark.sql.functions import (
    col, year, month, date_trunc, 
    sum as _sum, count as _count, round as _round, 
    current_timestamp
)
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
SILVER_ORDERS_PATH = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/delta/orders_silver"
GOLD_BASE_PATH     = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

GOLD_FACT_ORDERS_PATH       = f"{GOLD_BASE_PATH}/delta/fact_orders"
GOLD_DAILY_METRICS_PATH     = f"{GOLD_BASE_PATH}/delta/agg_daily_sales"
GOLD_CUSTOMER_METRICS_PATH  = f"{GOLD_BASE_PATH}/delta/agg_customer_metrics"

# COMMAND ----------

# CELL 2: Read Validated Silver Layer Data
silver_df = spark.read.format("delta").load(SILVER_ORDERS_PATH)
display(silver_df)

# COMMAND ----------

# CELL 3: Build Fact Table with Partition Keys (fact_orders)
# Extract partition attributes: order_year and order_month
fact_orders_df = (
    silver_df
    .withColumn("order_year", year(col("order_timestamp")))
    .withColumn("order_month", month(col("order_timestamp")))
    .withColumn("_gold_loaded_at", current_timestamp())
    .select(
        "order_id",
        "customer_id",
        "product_id",
        "amount",
        "status",
        "order_timestamp",
        "order_year",
        "order_month",
        "_gold_loaded_at"
    )
)

# Write Fact Table partitioned by order_year and order_month
(
    fact_orders_df.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("order_year", "order_month")
    .save(GOLD_FACT_ORDERS_PATH)
)

print(f"Fact Table successfully created at: {GOLD_FACT_ORDERS_PATH}")

# COMMAND ----------

# CELL 4: Build Aggregated Business Marts

# Mart A: Daily Sales & Order Volumes (For Executive Dashboards)
daily_sales_df = (
    fact_orders_df
    .filter(col("status") == "COMPLETED")
    .groupBy(date_trunc("day", col("order_timestamp")).alias("sale_date"))
    .agg(
        _count("order_id").alias("total_orders"),
        _round(_sum("amount"), 2).alias("total_revenue")
    )
    .withColumn("_calculated_at", current_timestamp())
)

(
    daily_sales_df.write
    .format("delta")
    .mode("overwrite")
    .save(GOLD_DAILY_METRICS_PATH)
)

# Mart B: Customer Lifetime Value / Spending Summary
customer_metrics_df = (
    fact_orders_df
    .groupBy("customer_id")
    .agg(
        _count("order_id").alias("lifetime_order_count"),
        _round(_sum("amount"), 2).alias("lifetime_spend")
    )
    .withColumn("_calculated_at", current_timestamp())
)

(
    customer_metrics_df.write
    .format("delta")
    .mode("overwrite")
    .save(GOLD_CUSTOMER_METRICS_PATH)
)

print("Business aggregate marts built successfully.")

# COMMAND ----------

# CELL 5: Delta Lake Performance Optimization (OPTIMIZE & Z-ORDER)

# 1. Compact small files into optimal chunks (~1GB) and co-locate related records
# Z-Ordering along high-cardinality join/filter predicates: customer_id & product_id
spark.sql(f"""
    OPTIMIZE delta.`{GOLD_FACT_ORDERS_PATH}`
    ZORDER BY (customer_id, product_id)
""")

# 2. Compact the customer aggregation table
spark.sql(f"""
    OPTIMIZE delta.`{GOLD_CUSTOMER_METRICS_PATH}`
""")

print("OPTIMIZE and Z-ORDER execution completed.")

# COMMAND ----------

# CELL 6: Verify Optimization Metrics and Pruning Plan

# Inspect Delta transaction operations history (shows numAddedFiles, numRemovedFiles)
display(spark.sql(f"DESCRIBE HISTORY delta.`{GOLD_FACT_ORDERS_PATH}`"))

# Check partition pruning and data skipping with an EXPLAIN query plan
fact_table = spark.read.format("delta").load(GOLD_FACT_ORDERS_PATH)
fact_table.createOrReplaceTempView("tv_fact_orders")

query = spark.sql("""
    SELECT 
        customer_id, 
        SUM(amount) AS total_spent
    FROM tv_fact_orders
    WHERE order_year = 2026 AND order_month = 3 AND customer_id = 'C001'
    GROUP BY customer_id
""")

# Output the physical Spark execution plan showing partition filters & pushdown
query.explain(True)
query.show()