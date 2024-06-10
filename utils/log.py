"""
@author Anh Tu Duong (aduong@fbk.eu)
@brief Utility functions for logging
@date 23-05-2024
"""

# Resolve paths
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Class Log
class Log:
    """!
    @brief This class contains utility functions for logging.
    """
    def info(text):
        out = (f"[INFO]\t{text}")
        print(out)
        return out

    def debug(text):
        out = (f"[DEBUG]\t{text}")
        print(out)
        return out

    def debug_highlight(text):
        out = (f"[DEBUG]\t{text}")
        print(out)
        return out

    def error(text):
        out = (f"[ERROR]\t{text}")
        print(out)
        return out

    def warning(text):
        out = (f"[WARNING]\t{text}")
        print(out)
        return out
    
    # TODO: implement log to file