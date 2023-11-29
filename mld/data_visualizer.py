import torch
import numpy as np
from mld.data.get_data import get_datasets
from mld.config import parse_args

cfg = parse_args(phase="demo")
cfg.FOLDER = cfg.TEST.FOLDER
cfg.Name = cfg.NAME

dataset = get_datasets(cfg, logger=None, phase="train")[0]

data_name = "Example_50_batch0_0"
caption   = "a person walks backward slowly."
data_path = "CL_test/"
motion_data = np.load("./" + data_path + data_name + ".npy")

from mld.data.humanml.utils.plot_script import plot_3d_motion
mp4path = str("./" + data_path + data_name + ".mp4")
plot_3d_motion(mp4path, joints=motion_data,  title=caption, fps=20)