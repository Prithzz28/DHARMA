# conftest.py — pytest configuration for the backend package
import sys
import os

# Ensure backend root and workspace root are on sys.path so all modules resolve correctly
# when running pytest from backend/ or the repository root.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

root_dir = os.path.dirname(backend_dir)
if root_dir not in sys.path:
    sys.path.insert(1, root_dir)
