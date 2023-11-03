from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
from torch.optim.adam import Adam
from PIL import Image


class NullInversion:    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):        
        timestep, next_timestep = min(timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999), timestep
        
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]
        
        beta_prod_t = 1 - alpha_prod_t
        
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        
        return next_sample
    
    def get_noise_pred_single(self, latents, mo_len, t, context):
        # noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        noise_pred = self.model.denoiser(
                            sample=latents,
                            timestep=t.to(latents.device),
                            encoder_hidden_states=context,
                            lengths=[mo_len]
                        )[0]
        
        return noise_pred

    def get_noise_pred(self, latents, mo_len, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        length_reverse = ([mo_len] * 2)
        
        if context is None:
            context = self.context
            
        guidance_scale = 1 if is_forward else self.guidance_scale
        
        noise_pred = self.model.denoiser(
                            sample=latents_input,
                            timestep=t.to(latents.device),
                            encoder_hidden_states=context,
                            lengths=length_reverse)[0]
        
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2motion(self, latents, motion_len, return_type='np'):
        feats_rst = self.model.vae.decode(latents, [motion_len])
        # print(feats_rst.shape)  # torch.Size([1, 60, 263])
        
        motion = self.model.feats2joints(feats_rst.detach().cpu())
        # print(motion.shape)     #torch.Size([1, 60, 22, 3])
        
        return motion

    @torch.no_grad()
    def motion2latent(self, motion, motion_len):
        with torch.no_grad():
            motion_data = motion.unsqueeze(0)
            latents, _ = self.model.vae.encode(motion_data, [motion_len])
            
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        # uncond_input = self.text_encoder.tokenizer(
        #     [""],
        #     padding="max_length",
        #     max_length=self.text_encoder.tokenizer.max_length,
        #     return_tensors="pt"
        # )
        # uncond_embeddings \
        #         = self.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        uncond_embeddings = self.text_encoder([""])   
        
        # text_input = self.text_encoder.tokenizer(
        #     [prompt],
        #     padding="max_length",
        #     max_length=self.text_encoder.tokenizer.max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # text_embeddings \
        #         = self.text_encoder(text_input.input_ids.to(self.model.device))[0]
        text_embeddings = self.text_encoder([prompt])
        
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, mo_len):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        
        all_latent = [latent]
        latent = latent.clone().detach()
        
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, mo_len, t, cond_embeddings)
                       
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, motion):
        mo_len = motion.shape[0]
        
        latent     = self.motion2latent(motion, mo_len)
        motion_rec = self.latent2motion(latent, mo_len)
        
        ddim_latents = self.ddim_loop(latent, mo_len)
        
        return motion_rec, ddim_latents

    def null_optimization(self, latents, mo_len, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, mo_len, t, cond_embeddings)
                            
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, mo_len, t, uncond_embeddings)
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                                             
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_item = loss.item()
                bar.update()               
                
                if loss_item < epsilon + i * 2e-5:
                    break
                
            for j in range(j + 1, num_inner_steps):
                bar.update()
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach().clone())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, mo_len, t, False, context)
                
        bar.close()
        
        return uncond_embeddings_list
    
    def invert(self, motion, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=True):
        self.init_prompt(prompt)
        
        motion_gt = torch.Tensor(motion).to(self.model.device)
        
        if verbose:
            print("##### DDIM inversion... #####")
        mo_len = motion_gt.shape[0]
        motion_rec, ddim_latents = self.ddim_inversion(motion_gt)
        
        # print(len(ddim_latents))
        
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, mo_len, num_inner_steps, early_stop_epsilon)
        
        return (motion_gt, motion_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model, cfg):
        self.model = model
        self.model.scheduler.set_timesteps(cfg.model.scheduler.num_inference_timesteps)
        
        self.text_encoder = self.model.text_encoder
        self.num_ddim_steps = cfg.scheduler.num_inference_timesteps
        self.guidance_scale = cfg.guidance_scale
        self.prompt = None
        self.context = None
        self.cfg = cfg
