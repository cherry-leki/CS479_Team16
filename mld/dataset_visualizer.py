import torch
from mld.data.get_data import get_datasets
from mld.config import parse_args

cfg = parse_args(phase="demo")
cfg.FOLDER = cfg.TEST.FOLDER
cfg.Name = cfg.NAME
    
dataset = get_datasets(cfg, logger=None, phase="train")[0]
train_dataset = dataset.train_dataset
print("####### Dataset Info. #######")
print("Dataset len: {}".format(len(train_dataset)))

# train dataset 데이터 리스트 (파일번호)
print(train_dataset.name_list)

# name list와 caption 대응 관계 확인
# print(train_dataset.name_list[0])
# print(train_dataset[0][2])
# print(train_dataset.name_list[1])
# print(train_dataset[1][2])
# print(caption_list[0])

sample     = train_dataset[0]
word_emb   = sample[0]
pos_onehot = sample[1]
caption    = sample[2]
joint_len  = sample[3]
motion     = sample[4]
motion_len = sample[5]
tokens     = sample[6]

print("####### Input motion Info. #######")
print(caption)
print(motion_len)
print("##################################")

from mld.data.humanml.utils.plot_script import plot_3d_motion
vis_motion  = dataset.feats2joints(torch.tensor(motion)).detach().cpu().numpy()
output_dir = "./results/humanml3d"
mp4path = str(output_dir + "/original.mp4")
# plot_3d_motion(mp4path, joints=vis_motion,  title=caption,  fps=20)