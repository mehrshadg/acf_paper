import glob
import numpy as np
from joblib import Parallel, delayed
from neuro_helper.abstract.map import TemplateMap
from neuro_helper.hcp.meg.generic import task_order
from neuro_helper.hcp.meg.storage import MEGLocalStorage, load_raw_file
from neuro_helper.measurement import calc_acz
from neuro_helper.generic import out_of, generate_long_data
from neuro_helper.generic import find_shared_subjects as fs_subjects
from neuro_helper.storage import ANYTHING


def do_a_file(file):
    print("Calculating ACZ on %s" % file)
    data, fs = load_raw_file(file)
    return calc_acz(data, n_job=5) / fs


def run_script(tpt: TemplateMap):
    from config import RAW_DATA_ROOT_DIR
    for task in task_order():
        storage = MEGLocalStorage(RAW_DATA_ROOT_DIR, tpt.name, task, ANYTHING)
        files_dict = storage.get_all_by_scan()
        for scan_id, file_infos in files_dict.items():
            output_file = out_of(f"megs-hcp-{task}.acz.rois-{tpt.name}.scan-{scan_id}.npy", False)
            subj_ids, files = list(zip(*file_infos))
            output = np.asarray(Parallel(n_jobs=30)(delayed(do_a_file)(file) for file in files))
            np.save(output_file, (task, scan_id, subj_ids, output))


def find_files(**kwargs):
    task = kwargs["task"]
    tpt = kwargs["template"]
    files = glob.glob(out_of(f"megs-hcp-{task}.acz.rois-{tpt.name}.scan-{ANYTHING}.npy", False))
    files.sort()
    return files


def prepare_file_content(ret_metric):
    if ret_metric:
        return lambda content: (content[1], content[2], content[3][:, 1:])
    else:
        return lambda content: (content[1], content[2])


def find_shared_subjects(tpt: TemplateMap, tasks, return_indices=False):
    return fs_subjects(find_files, prepare_file_content(False), tpt, tasks, return_indices)


def gen_long_data(tpt: TemplateMap):
    return generate_long_data(find_files, prepare_file_content(True), tpt, task_order(True))
