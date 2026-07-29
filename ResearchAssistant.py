import os

# Prevent duplicate OpenMP runtime crashes and thread deadlocks
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# FIX: Pre-import heavy C-extensions here so they load safely in the main Windows process
# This prevents the Streamlit worker threads from triggering an Access Violation.
import pyarrow
import pandas
import sentence_transformers

from interface import main

if __name__ == "__main__":
    main()