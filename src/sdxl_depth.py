import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation


class SDXLDepth():
    def __init__(self, device):
        self.device = device
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16).to(device)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe.enable_model_cpu_offload()

    def get_text_embeddings(self, prompt, negative_prompt=None):
        pass

    def img2img_step(self, prompt, depth_mask, negative_prompt=None,
                     guidance_scale=100, strength=0.5, num_inference_steps=50,
                     update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5,
                     intermediate_vis=False):
        depth_map = torch.nn.functional.interpolate(
            depth_mask,
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

        images = self.pipe(prompt=prompt, image=image,
                           negative_prompt=negative_prompt,
                           controlnet_conditioning_scale=0.5,
                           output_type="latent").images

        return images, []


def get_depth_map(image):
    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-hybrid-midas")

    image = feature_extractor(
        images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


if __name__ == "__main__":
    sd = SDXLDepth("cuda")
    image = load_image(
        "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    depth_image = get_depth_map(image)
    image = sd.img2img_step("spider man lecture, marvel movie character, photorealistic",
                            depth_mask=depth_image)

    depth_image.save("depth.jpg")
    image.save(f"stormtrooper.png")
