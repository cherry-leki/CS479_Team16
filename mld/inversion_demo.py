from cgi import test
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger

from inversion.null_text_inversion import NullInversion


def run_null_text_inv(cfg, model, prompt, mo_len, latent=None, uncond_embeddings=None, verbose=True):
    model.scheduler.set_timesteps(cfg.model.scheduler.num_inference_timesteps)
    timesteps = model.scheduler.timesteps.to(model.device)
    
    if uncond_embeddings is None:
        uncond_embedding = model.text_encoder([""])
        uncond_embeddings = [uncond_embedding] * cfg.model.scheduler.num_inference_timesteps
    
    cond_emb = model.text_encoder(prompt)
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latent] * 2))
            length_reverse = (mo_len * 2)
            
            encoder_hidden_states = torch.cat([uncond_embeddings[i], cond_emb])
            
            noise_pred = model.denoiser(
                sample=latent_model_input,
                timestep=t.to(model.device),
                encoder_hidden_states=encoder_hidden_states,
                lengths=length_reverse
            )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond \
                        + model.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
            latent = model.scheduler.step(noise_pred, t, latent).prev_sample
        
        latent = latent.permute(1, 0, 2)
        
        feats_rst = model.vae.decode(latent, mo_len)
        
    joint_rst = model.feats2joints(feats_rst.detach().cpu())
    
    return feats_rst, joint_rst


def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = cfg.NAME
    logger = create_logger(cfg, phase="demo")    

    # if cfg.DEMO.EXAMPLE:
    #     # Check txt file input
    #     # load txt
    #     from mld.utils.demo_utils import load_example_input

    #     text, length = load_example_input(cfg.DEMO.EXAMPLE)
    #     task = "Example"
    # else:
    #     # keyborad input
    #     task = "Keyborad_input"
    #     text = input("Please enter texts, none for random latent sampling:")
    #     length = input(
    #         "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
    #     )
    #     if text:
    #         motion_path = input(
    #             "Please enter npy_path for motion transfer, none for skip:")
    #     # text 2 motion
    #     if text and not motion_path:
    #         cfg.DEMO.MOTION_TRANSFER = False
    #     # motion transfer
    #     elif text and motion_path:
    #         # load referred motion
    #         joints = np.load(motion_path)
    #         frames = subsample(
    #             len(joints),
    #             last_framerate=cfg.DEMO.FRAME_RATE,
    #             new_framerate=cfg.DATASET.KIT.FRAME_RATE,
    #         )
    #         joints_sample = torch.from_numpy(joints[frames]).float()

    #         features = model.transforms.joints2jfeats(joints_sample[None])
    #         motion = xx
    #         # datastruct = model.transforms.Datastruct(features=features).to(model.device)
    #         cfg.DEMO.MOTION_TRANSFER = True

    #     # default lengths
    #     length = 200 if not length else length
    #     length = [int(length)]
    #     text = [text]

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    # total_time = time.time()
    model = get_model(cfg, dataset)

    # debugging
    # vae
    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    # mld_time = time.time()

    # sample
    # with torch.no_grad():
    ### Inversion test
    # Make test dataset
    test_data = dataset._sample_set[0]
    labels = ["word_emb", "pos_onehot", "caption", "joint_len", "motion", "motion_len", "tokens"]
    test_data = dict(zip(labels, test_data))
    test_motion_data = test_data["motion"]
    test_text_data   = test_data["caption"]
    test_motion_len  = test_motion_data.shape[0]
    print("####### Input motion Info. #######")
    print(test_text_data)
    print(test_motion_len)
    print("##################################")
    
    null_inversion = NullInversion(model, cfg)
    (motion_gt, motion_enc), x_t, uncond_embeddings = null_inversion.invert(test_motion_data,
                                                                            test_text_data)
    
    uncond_emb = str(output_dir / f"null_emb.txt")
    import numpy as np
    save_uncond_emb = [item[0][0].detach().cpu().numpy() for item in uncond_embeddings]
    np.savetxt(uncond_emb, np.asarray(save_uncond_emb), delimiter = ',')
    
    from mld.data.humanml.utils.plot_script import plot_3d_motion
    motion_gt  = model.feats2joints(motion_gt).detach().cpu().numpy()        
    motion_enc = motion_enc[0].detach().cpu().numpy()
    
    # print(motion_gt)
    # print(motion_rec.shape)
    
    # plot_3d_motion("./inversion/motion_gt.mp4",  joints=motion_gt,  title="motion_gt",  fps=20.0)
    # plot_3d_motion("./inversion/motion_rec.mp4", joints=motion_enc, title="motion_rec", fps=20.0)
    with torch.no_grad():        
        ddim_inv_motion, ddim_inv_joints = run_null_text_inv(cfg,
                                                             model,
                                                             test_text_data,
                                                             [test_motion_len],
                                                             latent=x_t,
                                                             uncond_embeddings=None)
        
        # run null text inversion
        null_inv_motion, null_inv_joints = run_null_text_inv(cfg,
                                                             model,
                                                             test_text_data,
                                                             [test_motion_len],
                                                             latent=x_t,
                                                             uncond_embeddings=uncond_embeddings)
        
        mp4path = str(output_dir / f"2D_gt.mp4")
        plot_3d_motion(mp4path, joints=motion_gt,  title="GT",  fps=20)
        mp4path = str(output_dir / f"2D_ddim_inv.mp4")
        plot_3d_motion(mp4path, joints=ddim_inv_joints[0].detach().cpu().numpy(), title="ddim_inv", fps=20)
        mp4path = str(output_dir / f"2D_null_inv.mp4")
        plot_3d_motion(mp4path, joints=null_inv_joints[0].detach().cpu().numpy(), title="null_inv", fps=20)
        
        # print(motion_gt.shape)
        # print(motion.shape)
        # print(joints.shape)
        
        # save npy data
        npypath = str(output_dir / f"gt.npy")
        np.save(npypath, motion_gt)
        
        npypath = str(output_dir / f"ddim_inv.npy")
        np.save(npypath, ddim_inv_joints[0].detach().cpu().numpy())
        
        npypath = str(output_dir / f"null_inv.npy")
        with open(npypath.replace("null_inv.npy", "text_prompt.txt"), "w") as text_file:
            text_file.write(str(test_motion_len) + " " + test_text_data)
        np.save(npypath, null_inv_joints[0].detach().cpu().numpy())
        
        logger.info(f"Motions are generated here:\n{output_dir}")



if __name__ == "__main__":
    main()
