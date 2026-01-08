"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import logging
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
print(project_root)
sys.path.append(project_root)
from src.models.transformers import CogVideoXTransformer3DModel, CogVideoXTransformer3DModelTuned
from src.pipelines.cogvideo import  CogVideoXDepthPipeline

from diffusers.utils import export_to_video, load_image, load_video


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}



import cv2
import PIL.Image
import PIL.ImageOps
from typing import Optional, Callable, Union

def load_image_from_video(
    video_path: Union[str], convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    Loads the first frame from a video file and returns it as a PIL Image.

    Args:
        video_path (`str`): Path to a local video file.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*): 
            A conversion method to apply to the image after loading it. 
            When set to `None`, the image will be converted to "RGB".

    Returns:
        `PIL.Image.Image`: The first video frame as a PIL Image.

    Raises:
        ValueError: If the video cannot be opened or the path is invalid.
    """
    if not isinstance(video_path, str) or not os.path.isfile(video_path):
        raise ValueError(f"Invalid video path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to read the first frame from the video.")

    # Convert OpenCV frame (BGR) to PIL image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(frame_rgb)
    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image

def smart_load_image(
    path: Union[str, PIL.Image.Image],
    convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    Loads an image from a file path. If the file is a video (.mp4), it loads the first frame as a PIL image.
    Otherwise, it uses the regular load_image function.

    Args:
        path (`str` or `PIL.Image.Image`): Path to the image or video file, or a PIL Image.
        convert_method (Callable, optional): A method to apply to the image after loading.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    if isinstance(path, str) and path.lower().endswith(".mp4"):
        return load_image_from_video(path, convert_method)
    else:
        return load_image(path, convert_method)



import os 
import cv2
import numpy as np

from safetensors.torch import load_file


import argparse
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import numpy as np
from utils.dc_utils import read_video_frames, save_video

def save_depth_video(batch_output, depth_filename, fps=30):



    first_frame = batch_output[0]


    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(depth_filename, fourcc, fps, (1360, 768), isColor=False)


    for frame in batch_output:

        depth_array = np.array(frame)


        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


        video_writer.write(depth_normalized)


    video_writer.release()
    print(f"Depth video saved to {depth_filename}")



import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


import os



def generate_video(
    prompts_path: str,
    model_path: str,
    sft_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None
    
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]

    if width is None or height is None:
        height, width = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution


    transformer = CogVideoXTransformer3DModelTuned(ofs_embed_dim = 512 if generate_type == "i2v" else None).from_pretrained(sft_path, subfolder="transformer", torch_dtype=dtype)

    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)


    pipe = CogVideoXDepthPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
    )


    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    

            
    with open(prompts_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if generate_type == "i2v":
                prompt, image_or_video_path = line.split('\t')
            else:
                prompt = line
            print(f"[{idx}] Prompt: {prompt}")
            filename = f"{idx + 1:05d}.mp4"
            video_path = os.path.join(output_path, filename)

            # 4. Generate the video frames based on the prompt.
            # `num_frames` is the Number of frames to generate.
            if generate_type == "i2v":
                image = smart_load_image(image_or_video_path)
                pipe_return = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    image=image,
                    # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
                    num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
                    num_inference_steps=num_inference_steps,  # Number of inference steps
                    num_frames=num_frames,  # Number of frames to generate
                    use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
                    process_image = True
                )
            elif generate_type == "t2v":

                pipe_return = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps,
                    # num_inference_steps=1,
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),
                )
            else:
                pipe_return = pipe(
                    height=height,
                    width=width,
                    prompt=prompt,
                    video=video,  # The path of the video to be used as the background of the video
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
                )
            video_generate = pipe_return.frames[0]

            depth_generate = pipe_return.depth_frames[0]
            
            os.makedirs(output_path, exist_ok=True)
            
            export_to_video(video_generate, video_path, fps=fps)

            name, ext = os.path.splitext(filename)

            frames = pipe_return.depth_output.squeeze().to(dtype=torch.float32).cpu().numpy()

            save_video(frames, output_path+'/'+ name + '_depth.mp4', fps=16, is_depths=True, grayscale=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompts_path", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="./CogVideoX1.5-5B", help="Path of the pre-trained model use"
    )
    parser.add_argument(
        "--sft_path", type=str, default="./checkpoints", help="Path of the finetuned model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./outputs", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=0, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompts_path=args.prompts_path,
        model_path=args.model_path,
        sft_path=args.sft_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )