from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs, listdir


def convert_device_name(framework):
    """Convert device to either cpu or cuda."""
    gpu_names = ["gpu", "cuda"]
    cpu_names = ["cpu"]
    if framework not in cpu_names + gpu_names:
        raise KeyError("the device should either "
                       "be cuda or cpu but got {}".format(framework))
    if framework in gpu_names:
        return "cuda"
    else:
        return "cpu"


def convert_framework_name(framework):
    """Convert framework to either tf or torch."""
    tf_names = ["tf", "tensorflow", "TF"]
    torch_names = ["torch", "pytorch", "PyTorch"]
    if framework not in tf_names + torch_names:
        raise KeyError("the framework should either "
                       "be tf or torch but got {}".format(framework))
    if framework in tf_names:
        return "tf"
    else:
        return "torch"

def make_dir(folder_name):
    """Create a directory.
    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)