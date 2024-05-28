import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base_libs import *

def check_folder_exist(*args, **kwargs):
    if len(args) != 0:
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    if len(kwargs) != 0:
        for path in kwargs.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)


