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

def delete_file_cronj(path_folder, stime, format_time='%Y-%m-%d_%H-%M.jpg', ratio_delete=0.6):
	while True:
		time.sleep(stime)
		list_f = os.listdir(path_folder)
		list_f_sorted = sorted(list_f, key=lambda t: datetime.strptime(t, format_time))
		num_delete = int(len(list_f_sorted)*ratio_delete)
		for i in range(num_delete):
			os.remove(os.path.join(path_folder,list_f_sorted[i]))
