import torch
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    jtr = np.load("Jtr.npy")
    trans = np.load("translation.npy")
    
    trans = trans[:, None]
    trans = trans[..., (1, 0, 2)]
    trans[..., 0] *= -1

    # debugging
    from mld.data.humanml.utils.plot_script import plot_3d_motion
    
    gt_mp4path        = str(f"2D_gt.mp4")
    plot_3d_motion(gt_mp4path, joints=jtr + trans,  title="test",  fps=20)

    # # combine
    # from moviepy.editor import VideoFileClip, clips_array
    # gt_mp4       = VideoFileClip(gt_mp4path)
    # final_clip = clips_array([[gt_mp4]])
    # final_clip.write_videofile(str(output_dir / f"2D_all.mp4"), fps=20)
    


if __name__ == "__main__":
    main()
