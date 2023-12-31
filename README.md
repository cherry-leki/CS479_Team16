# CS479_Team16
Text-guided Motion Editing with Inversion Technique


# Set-up
Our model is based on MLD, so please follow the public MLD GitHub's instructions <b>Quick Start</b>. \
https://github.com/ChenFengYe/motion-latent-diffusion

We provide two ways to prepare input motion samples.
* Sample set and training set provided by HumanML3D
* Input motion created in MLD through input text prompt

Choosing which input method to use and setting the edited text prompt can be done from ```line 110``` of ```mld/control_demo.py```.
<pre>
  <code>
    * input_motion_info -> the method of how to prepare input motion data
    * input_text        -> the input text prompt related to the input motion data
    * input_mo_len      -> the length of generated motion
    * convert_text_list -> the list of edited text prompts
  </code>
</pre>


After setting up the HumanML3D dataset and MLD environment, run the code with the following command.

<pre>
  <code>
  python control_demo.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml
  </code>
</pre>



# Baseline codes
Our code is implemented based on belows.
* [MLD: Motion Latent Diffusion Models](https://github.com/ChenFengYe/motion-latent-diffusion)
* [Null-Text Inversion for Editing Real Images](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)


<pre>
  <code>
  @inproceedings{chen2023executing,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18000--18010},
  year={2023}
  }
  </code>
</pre>

<pre>
  <code>
  @article{mokady2022null,
  title={Null-text Inversion for Editing Real Images using Guided Diffusion Models},
  author={Mokady, Ron and Hertz, Amir and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2211.09794},
  year={2022}
  </code>
</pre>
