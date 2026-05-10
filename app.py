"""
Root-level entry point for Streamlit app.
This file allows running: streamlit run app.py
from the project root directory.
"""

import sys
from pathlib import Path

# Import and run the UI app from ui/app.py
# The ui/app.py module runs automatically when imported
import ui.app
