import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline


class SDXL:
    def __init__(self, device):
        self.device = device

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet, variant="fp16", use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)

        pipe.enable_model_cpu_offload()

    def img2img_step(self, prompt, negative_prompt, image, depth_mask):
        depth_mask = torch.nn.functional.interpolate(
            depth_mask.unsqueeze(1), size=(1024, 1024), mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_mask = (depth_mask - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_mask] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

        images = self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                           image=image, control_image=depth_mask,
                           strength=0.99, num_inference_steps=50,
                           controlnet_conditioning_scale=0.5,
                           output_type="np").images

        return images[0], []
