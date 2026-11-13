"""Resilient HDFS & Local Storage Bridge for Big Data Analytics.

Provides seamless reading/writing to HDFS via WebHDFS REST API and PyArrow,
with automatic fallback and synchronization to local lakehouse directory.
"""
from __future__ import annotations

import io
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Ensure project root is in sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HDFS_Bridge")


def is_hdfs_rpc_alive(host: str = config.HDFS_HOST, port: int = config.HDFS_PORT, timeout: float = 2.0) -> bool:
    """Check if HDFS RPC port is responding."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_webhdfs_alive(base_url: str = config.WEBHDFS_BASE_URL, timeout: float = 2.0) -> bool:
    """Check if WebHDFS HTTP REST API is reachable."""
    try:
        resp = requests.get(f"{base_url}/?op=LISTSTATUS", timeout=timeout)
        return resp.status_code in (200, 401, 403)
    except Exception:
        return False


class HDFSClient:
    """Unified client for writing and reading Parquet tables to HDFS & local storage."""

    def __init__(
        self,
        base_uri: str = config.HDFS_BASE_URI,
        webhdfs_url: str = config.WEBHDFS_BASE_URL,
        local_root: Path = config.LOCAL_LAKE_DIR,
    ):
        self.base_uri = base_uri.rstrip("/")
        self.webhdfs_url = webhdfs_url.rstrip("/")
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._check_connection()

    def _check_connection(self) -> None:
        self.rpc_available = is_hdfs_rpc_alive(timeout=0.5)
        self.webhdfs_available = is_webhdfs_alive(timeout=0.5)
        if self.rpc_available or self.webhdfs_available:
            logger.info("✓ Connected to HDFS cluster (RPC: %s, WebHDFS: %s)", self.rpc_available, self.webhdfs_available)
        else:
            logger.warning("! HDFS cluster offline. Operating with local filesystem mirror at '%s'", self.local_root)

    def _resolve_local_path(self, hdfs_path: str) -> Path:
        """Map HDFS path (e.g. '/lake/gold/market_features_1h') to local mirror."""
        clean = hdfs_path.strip("/").replace("lake/", "")
        local_path = self.local_root / clean
        if not local_path.suffix:
            local_path = local_path.with_suffix(".parquet")
        return local_path

    def write_parquet(
        self,
        df: pd.DataFrame,
        hdfs_path: str,
        sync_local: bool = True,
    ) -> bool:
        """Write a DataFrame as Parquet to HDFS and mirror locally."""
        local_path = self._resolve_local_path(hdfs_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Always save locally first as reliable backup & fast read
        table = pa.Table.from_pandas(df)
        pq.write_table(table, local_path, compression="snappy")
        logger.info("  [Local] Written %d rows to '%s'", len(df), local_path)

        # 2. Upload to HDFS if available via WebHDFS
        hdfs_written = False
        if self.webhdfs_available:
            try:
                buffer = io.BytesIO()
                pq.write_table(table, buffer, compression="snappy")
                buffer.seek(0)

                # WebHDFS Create file step 1
                clean_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
                url = f"{self.webhdfs_url}{clean_path}?op=CREATE&overwrite=true"
                r = requests.put(url, allow_redirects=False, timeout=10)
                if r.status_code == 307:
                    datanode_url = r.headers["Location"]
                    # Step 2: Upload data to DataNode
                    r_upload = requests.put(
                        datanode_url,
                        data=buffer.getvalue(),
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=30,
                    )
                    if r_upload.status_code in (200, 201):
                        logger.info("  [HDFS] Uploaded %d rows to 'hdfs://%s'", len(df), clean_path)
                        hdfs_written = True
                    else:
                        logger.warning("  [HDFS] DataNode write failed with status %d", r_upload.status_code)
                else:
                    logger.warning("  [HDFS] NameNode redirect failed: %s", r.text)
            except Exception as e:
                logger.warning("  [HDFS] Upload to HDFS failed (%s). Local copy is intact.", e)

        return hdfs_written or sync_local

    def read_parquet(self, hdfs_path: str) -> pd.DataFrame:
        """Read a DataFrame from HDFS or local mirror fallback."""
        clean_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
        local_path = self._resolve_local_path(hdfs_path)

        # 1. Try reading directly from HDFS if online
        if self.webhdfs_available:
            try:
                url = f"{self.webhdfs_url}{clean_path}?op=OPEN"
                resp = requests.get(url, stream=True, timeout=10)
                if resp.status_code == 200:
                    buffer = io.BytesIO(resp.content)
                    df = pd.read_parquet(buffer)
                    logger.info("  [HDFS] Successfully read %d rows from '%s'", len(df), clean_path)
                    return df
            except Exception as e:
                logger.warning("  [HDFS] Read from WebHDFS failed (%s), falling back to local mirror.", e)

        # 2. Fallback to local mirror
        if local_path.exists():
            df = pd.read_parquet(local_path)
            logger.info("  [Local] Successfully read %d rows from '%s'", len(df), local_path)
            return df

        raise FileNotFoundError(f"Dataset not found in HDFS ('{clean_path}') or local mirror ('{local_path}').")


# Global default client
default_hdfs_client = HDFSClient()
