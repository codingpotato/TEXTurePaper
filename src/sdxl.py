import numpy as np
import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image


class SDXL:
    def __init__(self, device):
        self.device = device
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe_img2img = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe.enable_model_cpu_offload()

    def __call__(self, image, depth_mask, prompt, negative_prompt=None):
        print(depth_mask)
        depth_mask = torch.nn.functional.interpolate(
            depth_mask,
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_mask = (depth_mask - depth_min) / (depth_max - depth_min)
        depth_mask = torch.cat([depth_mask] * 3, dim=1)

        depth_mask = depth_mask.permute(0, 2, 3, 1).cpu().numpy()[0]
        depth_mask = Image.fromarray(
            (depth_mask * 255.0).clip(0, 255).astype(np.uint8))

        if image is not None:
            image_min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
            image_max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
            image = (image - image_min) / (image_max - image_min)
            image = torch.nn.functional.interpolate(
                image, size=(1024, 1024), mode="bicubic", align_corners=False,
            )

            images = self.pipe_img2img(prompt=prompt, image=image,
                                       control_image=depth_mask,
                                       negative_prompt=negative_prompt,
                                       num_inference_steps=50,
                                       strength=1.0,
                                       controlnet_conditioning_scale=0.5,
                                       output_type="np").images
        else:
            images = self.pipe(prompt=prompt, image=depth_mask,
                               negative_prompt=negative_prompt,
                               num_inference_steps=50,
                               controlnet_conditioning_scale=0.5,
                               output_type="np").images

        return images[0], []
