import enum
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader

from mld.config import parse_args
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
    
    with torch.no_grad():
        cond_emb_change = 10
        for i, t in enumerate(timesteps):
            if len(prompt) > 1:
                if i < cond_emb_change:
                    if verbose: print(str(i), t.item(), prompt[1])
                    cond_emb = model.text_encoder(prompt[1])
                else:
                    if verbose: print(str(i), t.item(), prompt[0])
                    cond_emb = model.text_encoder(prompt[0])
            else: 
                cond_emb = model.text_encoder(prompt)
            
            latent_model_input = (torch.cat([latent] * 2))
            length_reverse = (mo_len * 2)
            
            encoder_hidden_states = torch.cat([uncond_embeddings[i].expand(*cond_emb.shape), cond_emb])
            
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

    if False:
        if cfg.DEMO.EXAMPLE:
            # Check txt file input
            # load txt
            from mld.utils.demo_utils import load_example_input

            text, length = load_example_input(cfg.DEMO.EXAMPLE)
            task = "Example"
        else:
            # keyborad input
            task = "Keyborad_input"
            text = input("Please enter texts, none for random latent sampling:")
            length = input(
                "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
            )
            if text:
                motion_path = input(
                    "Please enter npy_path for motion transfer, none for skip:")
            # text 2 motion
            if text and not motion_path:
                cfg.DEMO.MOTION_TRANSFER = False
            # motion transfer
            elif text and motion_path:
                # load referred motion
                joints = np.load(motion_path)
                frames = subsample(
                    len(joints),
                    last_framerate=cfg.DEMO.FRAME_RATE,
                    new_framerate=cfg.DATASET.KIT.FRAME_RATE,
                )
                joints_sample = torch.from_numpy(joints[frames]).float()

                features = model.transforms.joints2jfeats(joints_sample[None])
                motion = xx
                # datastruct = model.transforms.Datastruct(features=features).to(model.device)
                cfg.DEMO.MOTION_TRANSFER = True

            # default lengths
            length = 200 if not length else length
            length = [int(length)]
            text = [text]

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="train")[0]

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


    ### Inversion
    # Make test dataset
    test_data = dataset._sample_set[2]
    labels = ["word_emb", "pos_onehot", "caption", "joint_len", "motion", "motion_len", "tokens"]
    test_data = dict(zip(labels, test_data))
    # test_motion_data = test_data["motion"]
    # test_text_data   = test_data["caption"]
    # test_motion_len  = test_motion_data.shape[0]
        
    # convert_text = test_text_data.replace("right", "both")       
    
    # print("####### Input motion Info. #######")
    # print(test_text_data)
    # print(test_motion_len)
    # print("Convert => ", convert_text)
    # print("##################################")
    
    input_text   = "a person steps to the right"
    # input_text   = "a person does a short jump"
    
    test_motion_len = 80
    batch = {"length": [test_motion_len], "text": [input_text]}
    input_motion = model(batch)    
    input_motion = input_motion[1][0]
              
    convert_text_list = [
                        "a person steps to the diagonal",
                        "a person steps to the front",
                        "a person steps to the back",
                        "a person steps to the left",
                        ]
    # convert_text_list = [
    #                     "a person does a middle jump",
    #                     "a person does a long jump",
    #                     "a person does a jump forward",
    #                     ]
    
    print("####### Input motion Info. #######")
    print(input_text)
    print(test_motion_len)
    print("Convert => ", convert_text_list)
    print("##################################")
    
    null_inversion = NullInversion(model, cfg)
    (motion_gt, motion_rec), x_t, uncond_embeddings = null_inversion.invert(input_motion,
                                                                            input_text)
    
    # uncond_emb = str(output_dir / f"null_emb.txt")
    import numpy as np
    save_uncond_emb = [item[0][0].detach().cpu().numpy() for item in uncond_embeddings]
    # np.savetxt(uncond_emb, np.asarray(save_uncond_emb), delimiter = ',')
    np.save(str(output_dir / f"null_emb.npy"), save_uncond_emb)
    np.save(str(output_dir / f'x_t.npy'), x_t.detach().cpu().numpy())   
    
    # run null text inversion
    with torch.no_grad():
        # run null text inversion
        ddim_inv_motion, null_inv_joints = run_null_text_inv(cfg,
                                                            model,
                                                            [input_text],
                                                            [test_motion_len],
                                                            latent=x_t.clone(),
                                                            uncond_embeddings=uncond_embeddings,
                                                            verbose=False)
        edited_joints_list = []
        for convert_text in convert_text_list:
            # ddim_inv_motion, ddim_inv_joints = run_null_text_inv(cfg,
            #                                                     model,
            #                                                     [input_text, convert_text],
            #                                                     [test_motion_len],
            #                                                     latent=x_t.clone(),
            #                                                     uncond_embeddings=None,
            #                                                     verbose=False)            
            
            # run motion editing with null text inversion
            edited_motion,   edited_joints   = run_null_text_inv(cfg,
                                                                model,
                                                                [input_text, convert_text],
                                                                [test_motion_len],
                                                                latent=x_t.clone(),
                                                                uncond_embeddings=uncond_embeddings,
                                                                verbose=False)
            edited_joints_list.append([convert_text, edited_joints])
               
        
        # debugging
        from mld.data.humanml.utils.plot_script import plot_3d_motion
        motion_gt  = model.feats2joints(motion_gt).detach().cpu().numpy()        
        motion_rec = motion_rec[0].detach().cpu().numpy()
        
        # batch = {"length": [test_motion_len], "text": [convert_text]}
        # gen_noinv, _  = model(batch)
        
        gt_mp4path        = str(output_dir / f"2D_gt.mp4")
        plot_3d_motion(gt_mp4path,        joints=motion_gt,  title=input_text,  fps=20)
        # rec_mp4path       = str(output_dir / f"2D_rec.mp4")
        # plot_3d_motion(rec_mp4path,     joints=motion_rec, title="vae_rec", fps=20)
        # gen_noinv_mp4path = str(output_dir / f"2D_gen_noinv.mp4")
        # plot_3d_motion(gen_noinv_mp4path, joints=gen_noinv[0].detach().cpu().numpy(), title=convert_text, fps=20)
        # ddiminv_mp4path   = str(output_dir / f"2D_ddim_inv.mp4")
        # plot_3d_motion(ddiminv_mp4path,   joints=ddim_inv_joints[0].detach().cpu().numpy(), title="ddim_inv", fps=20)
        nullinv_mp4path   = str(output_dir / f"2D_null_inv.mp4")
        plot_3d_motion(nullinv_mp4path,   joints=null_inv_joints[0].detach().cpu().numpy(), title="null_inv", fps=20)
        for i, [convert_text, edited_joints] in enumerate(edited_joints_list):
            edited_mp4path   = str(output_dir / f"2D_edited_{i}.mp4")
            plot_3d_motion(edited_mp4path,    joints=edited_joints[0].detach().cpu().numpy(), title=convert_text, fps=20)
            edited_joints_list[i].append(edited_mp4path)
        
        # combine
        from moviepy.editor import VideoFileClip, clips_array
        gt_mp4       = VideoFileClip(gt_mp4path)
        # rec_mp4      = VideoFileClip(rec_mp4path)
        # gen_noiv_mp4 = VideoFileClip(gen_noinv_mp4path)
        # ddiminv_mp4  = VideoFileClip(ddiminv_mp4path)
        nullinv_mp4  = VideoFileClip(nullinv_mp4path)
        edited_mp4_list = [VideoFileClip(edited_mp4path) for [_, _, edited_mp4path] in edited_joints_list]
        # edited_mp4   = VideoFileClip(edited_mp4path)
        
        # final_clip = clips_array([[gt_mp4, rec_mp4, gen_noiv_mp4, ddiminv_mp4, nullinv_mp4]])
        # final_clip = clips_array([[gt_mp4, gen_noiv_mp4, ddiminv_mp4, nullinv_mp4, edited_mp4]])
        final_clip = clips_array([[gt_mp4, nullinv_mp4, *edited_mp4_list]])
        final_clip.write_videofile(str(output_dir / f"2D_all.mp4"), fps=20)
        
        
        # save npy data
        npypath = str(output_dir / f"gt.npy")
        np.save(npypath, motion_gt)
        
        # npypath = str(output_dir / f"gen_noinv.npy")
        # np.save(npypath, gen_noinv[0].detach().cpu().numpy())
        
        # npypath = str(output_dir / f"ddim_inv.npy")
        # np.save(npypath, ddim_inv_joints[0].detach().cpu().numpy())
        
        npypath = str(output_dir / f"null_inv.npy")
        np.save(npypath, null_inv_joints[0].detach().cpu().numpy())
        
        for i, [_, edited_joints, _] in enumerate(edited_joints_list):
            npypath = str(output_dir / f"edited_{i}.npy")
            np.save(npypath, edited_joints[0].detach().cpu().numpy())
        
        # text file for motion length and text
        with open(str(output_dir / f"text_prompt.txt"), "w") as text_file:
            text_file.write(str(test_motion_len) + " " + input_text)
        
        logger.info(f"Motions are generated here:\n{output_dir}")



if __name__ == "__main__":
    main()
