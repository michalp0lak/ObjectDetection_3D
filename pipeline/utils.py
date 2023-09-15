import os
import re
import time
from datetime import datetime

def latest_ckpt(train_ckpt_dir):

    def atoi( text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    files = os.listdir(train_ckpt_dir)

    if not files: 
        return None
    else:

        ckpt_list = [f for f in files if f.endswith('.pth') and 'ckpt_best' not in f]

        if len(ckpt_list) == 0:
            return None

        ckpt_list.sort(key = natural_keys)
        ckpt_name = ckpt_list[-1]

        return os.path.join(train_ckpt_dir, ckpt_name)
       
def resume_training_process(output_path):

    trials = []
    versions = []
    folders = os.listdir(output_path)

    if len(folders) == 0:
        return []

    for ver in folders:
        sp0 = ver.split('_')
        if len(sp0) == 2:
            sp1 = sp0[1].split('-')
            if (len(sp1) == 6) and bool(datetime.strptime(sp0[1], "%Y-%m-%d-%H-%M-%S")):
                versions.append(ver)

    for fl in versions:

        trials.append((fl, time.mktime(datetime.strptime(fl.split('_')[0], 
                       "%Y-%m-%d-%H-%M-%S").timetuple())))
    
    sorted_trials = sorted(trials, key=lambda tup: tup[1])
    
    return sorted_trials[-1][0]