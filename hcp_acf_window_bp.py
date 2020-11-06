import glob
import numpy as np
from joblib import Parallel, delayed
from neuro_helper.entity import TemplateName, Space
from neuro_helper.hcp.meg import MEGLocalStorage, load_raw_file, task_order
from neuro_helper.measurement import calc_a_acw
from neuro_helper.statistics import fir_filter_data
from neuro_helper.generic import out_of, generate_long_data
from neuro_helper.generic import find_shared_subjects as fs_subjects
from neuro_helper.storage import ANYTHING
import config


def do_a_file(file):
    print("Calculating ACW BP on %s" % file)
    data, fs = load_raw_file(file)
    data, freq_l, freq_h = fir_filter_data(data, fs, max_freq_low=8, min_freq_high=12, pass_type="bp")
    return np.asarray(Parallel(n_jobs=5)(delayed(calc_a_acw)(ts) for ts in data)) / fs


def run_script(tpt_name: TemplateName):
    for task in task_order():
        storage = MEGLocalStorage(config.RAW_DATA_ROOT_DIR, tpt_name, task, ANYTHING)
        files_dict = storage.get_all_by_scan()
        for scan_id, file_infos in files_dict.items():
            output_file = out_of(f"megs-hcp-alpha-{task}.acw.rois-{tpt_name}.scan-{scan_id}.npy", False)
            subj_ids, files = list(zip(*file_infos))
            output = np.asarray(Parallel(n_jobs=30)(delayed(do_a_file)(file) for file in files))
            np.save(output_file, (task, scan_id, subj_ids, output))


def find_files(**kwargs):
    task = kwargs["task"]
    template_lbl = kwargs["template_name"]
    files = glob.glob(out_of(f"megs-hcp-alpha-{task}.acw.rois-{template_lbl}.scan-*.npy", False))
    files.sort()
    return files


def prepare_file_content(ret_metric):
    if ret_metric:
        return lambda content: (content[1], content[2], content[3][:, 1:])
    else:
        return lambda content: (content[1], content[2])


def find_shared_subjects(template_name, tasks, return_indices=False):
    return fs_subjects(find_files, prepare_file_content(False), template_name, Space.K32, tasks, return_indices)


def gen_long_data(template_name):
    return generate_long_data(find_files, prepare_file_content(True), template_name, Space.K32)


if __name__ == "__main__":
    run_script(TemplateName.COLE_360)
