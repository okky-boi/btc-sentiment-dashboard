"""Centralized Configuration for Big Data Lakehouse & TFT Inference Pipeline.

Defines HDFS/local paths, the strict 54-column schema specification for
dataset_1h.parquet, medallion layer targets, and TFT model parameters.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------
# Paths & Environments
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Input files
DATASET_PARQUET = BASE_DIR / "dataset_1h.parquet"
MODEL_CKPT_DIR = BASE_DIR / "models" / "current"
MODEL_CKPT_PATH = MODEL_CKPT_DIR / "best.ckpt"

# Local Lakehouse mirror / fallback directory
LOCAL_LAKE_DIR = BASE_DIR / "lake"
LOCAL_BRONZE_DIR = LOCAL_LAKE_DIR / "bronze"
LOCAL_SILVER_DIR = LOCAL_LAKE_DIR / "silver"
LOCAL_GOLD_DIR = LOCAL_LAKE_DIR / "gold"
LOCAL_SERVING_DIR = LOCAL_LAKE_DIR / "serving"

# HDFS Configuration
HDFS_HOST = os.environ.get("HDFS_HOST", "localhost")
HDFS_PORT = int(os.environ.get("HDFS_PORT", "8020"))
HDFS_WEBHDFS_PORT = int(os.environ.get("HDFS_WEBHDFS_PORT", "9870"))
HDFS_BASE_URI = f"hdfs://{HDFS_HOST}:{HDFS_PORT}"
WEBHDFS_BASE_URL = f"http://{HDFS_HOST}:{HDFS_WEBHDFS_PORT}/webhdfs/v1"

# Medallion HDFS Endpoints
HDFS_BRONZE_DIR = "/lake/bronze"
HDFS_SILVER_DIR = "/lake/silver"
HDFS_GOLD_TABLE = "/lake/gold/market_features_1h"

# Serving Layer Endpoints
HDFS_SERVING_PRICE_ACTUAL = "/lake/serving/price_actual"
HDFS_SERVING_SENTIMENT = "/lake/serving/sentiment_hourly"
HDFS_SERVING_GRAPH_EDGES = "/lake/serving/graph_edges"
HDFS_SERVING_GRAPH_NODES = "/lake/serving/graph_nodes"
HDFS_SERVING_PRICE_FORECAST = "/lake/serving/price_forecast"

# --------------------------------------------------------------------------
# 54-Column Schema Definition (dataset_1h.parquet)
# --------------------------------------------------------------------------
COLUMN_GROUPS: Dict[str, List[str]] = {
    # 1. Pengenal Waktu (5)
    "TIME_INDEX": [
        "datetime",
        "time_idx",
        "series_id",
        "hour",
        "day_of_week",
    ],
    # 2. Binance Level 1 (OHLCV) (9)
    "L1_OHLCV": [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "taker_buy_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_ratio",
    ],
    # 3. Binance Derivatif & Order Book L2 (12)
    "L2_DERIVATIVES": [
        "Open_Interest",
        "Funding_Rate",
        "is_funding_event",
        "Order_Book_Imbalance",
        "Jarak_Resistance",
        "Jarak_Support",
        "n_snapshots",
        "snapshot_time",
        "wall_asymmetry",
        "oi_change_1h",
        "oi_change_24h",
        "funding_z",
    ],
    # 4. Sentimen FinBERT Reddit (11)
    "SENTIMENT": [
        "prob_positive",
        "prob_neutral",
        "prob_negative",
        "num_posts",
        "low_conf_share",
        "sent_carried_forward",
        "sent_neutral_default",
        "sent_gap_hours",
        "sent_net",
        "sent_change_24h",
        "log_post_count",
    ],
    # 5. Indikator Ketersediaan & Imputasi Data (9)
    "IMPUTATION_FLAGS": [
        "is_imputed_ohlcv",
        "available_ohlcv",
        "is_imputed_oi",
        "available_oi",
        "is_imputed_l2",
        "available_l2",
        "is_imputed_sent",
        "available_sent",
        "is_imputed_any",
    ],
    # 6. Volatilitas & Return Turunan (8)
    "DERIVED_DYNAMICS": [
        "log_rv_24h",
        "log_rv_168h",
        "log_rv_720h",
        "ret_1h",
        "ret_24h",
        "hl_range",
        "body",
        "obi_change",
    ],
}

ALL_54_COLUMNS: List[str] = [col for group in COLUMN_GROUPS.values() for col in group]

# Expected basic types for validation
EXPECTED_DTYPES: Dict[str, str] = {
    "datetime": "datetime64",
    "time_idx": "int",
    "series_id": "string",
    "hour": "string",
    "day_of_week": "string",
    "Open": "float",
    "High": "float",
    "Low": "float",
    "Close": "float",
    "Volume": "float",
    "taker_buy_volume": "float",
    "quote_volume": "float",
    "trade_count": "int",
    "taker_buy_ratio": "float",
    "Open_Interest": "float",
    "Funding_Rate": "float",
    "is_funding_event": "bool",
    "Order_Book_Imbalance": "float",
    "Jarak_Resistance": "float",
    "Jarak_Support": "float",
    "n_snapshots": "float",
    "snapshot_time": "datetime64",
    "wall_asymmetry": "float",
    "oi_change_1h": "float",
    "oi_change_24h": "float",
    "funding_z": "float",
    "prob_positive": "float",
    "prob_neutral": "float",
    "prob_negative": "float",
    "num_posts": "int",
    "low_conf_share": "float",
    "sent_carried_forward": "bool",
    "sent_neutral_default": "bool",
    "sent_gap_hours": "int",
    "sent_net": "float",
    "sent_change_24h": "float",
    "log_post_count": "float",
    "is_imputed_ohlcv": "bool",
    "available_ohlcv": "bool",
    "is_imputed_oi": "bool",
    "available_oi": "bool",
    "is_imputed_l2": "bool",
    "available_l2": "bool",
    "is_imputed_sent": "bool",
    "available_sent": "bool",
    "is_imputed_any": "bool",
    "log_rv_24h": "float",
    "log_rv_168h": "float",
    "log_rv_720h": "float",
    "ret_1h": "float",
    "ret_24h": "float",
    "hl_range": "float",
    "body": "float",
    "obi_change": "float",
}

# --------------------------------------------------------------------------
# Serving Schemas
# --------------------------------------------------------------------------
PRICE_ACTUAL_COLUMNS = [
    "datetime",
    "time_idx",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "log_rv_24h",
    "ret_1h",
]

SENTIMENT_HOURLY_COLUMNS = [
    "datetime",
    "prob_positive",
    "prob_neutral",
    "prob_negative",
    "sent_net",
    "num_posts",
]

GRAPH_EDGE_COLUMNS = [
    "source",
    "target",
    "weight",
    "type",
    "lag",
    "p_value",
]

PRICE_FORECAST_COLUMNS = [
    "datetime_target",
    "time_idx",
    "q10",
    "q50",
    "q90",
]

# --------------------------------------------------------------------------
# Temporal Fusion Transformer (TFT) Modeling Parameters
# --------------------------------------------------------------------------
TFT_LOOKBACK_HOURS = 168       # 7 days lookback window
TFT_HORIZON_HOURS = 24         # 24 hours prediction horizon
TFT_QUANTILES = [0.1, 0.5, 0.9]

TFT_BASE_OBSERVED = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Order_Book_Imbalance",
    "Jarak_Resistance",
    "Jarak_Support",
    "Open_Interest",
    "Funding_Rate",
    "prob_positive",
    "prob_neutral",
    "prob_negative",
]

TFT_ENGINEERED_EXT = [
    "ret_1h",
    "ret_24h",
    "hl_range",
    "body",
    "taker_buy_ratio",
    "log_rv_24h",
    "obi_change",
    "wall_asymmetry",
    "oi_change_1h",
    "oi_change_24h",
    "funding_z",
    "sent_net",
    "sent_change_24h",
    "log_post_count",
]

TFT_KNOWN_CATEGORICALS = ["hour", "day_of_week"]
TFT_TARGET = "Close"

