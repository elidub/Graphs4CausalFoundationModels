"""
High memory benchmark configuration - for memory-intensive workloads
"""

BATCH_SIZE = 512
NUM_WORKERS = 8
PREFETCH_FACTOR = 8
NUM_BATCHES = 5000
WARMUP = 500
SHUFFLE = True

# HTCondor job settings
REQUEST_CPUS = 8
REQUEST_MEMORY = "64GB"
REQUEST_DISK = "50GB"

# Job naming
CONFIG_NAME = "high_memory"
DESCRIPTION = "High memory benchmark for memory-intensive workload testing"
