import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from torchvision import transforms


class SDXL:
    def __init__(self, device):
        self.device = device
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16
        ).to(device)

        self.sdxl = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

        self.sdxl_img2img = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

        self.sdxl_inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=controlnet,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)
        self.sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.sdxl_inpaint.text_encoder_2,
            vae=self.sdxl_inpaint.vae,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

    def txt2img(self, prompt, depth_mask):
        depth_mask = self.preprocess_depth_mask(depth_mask)
        image = self.sdxl(prompt, image=depth_mask,
                          num_inference_steps=30,
                          controlnet_conditioning_scale=0.5,
                          output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)

    def img2img(self, prompt, image, depth_mask):
        depth_mask = self.preprocess_depth_mask(depth_mask)
        image = self.sdxl_img2img(prompt, image=image, control_image=depth_mask,
                                  num_inference_steps=30,
                                  controlnet_conditioning_scale=0.5,
                                  output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)

    def inpaint(self, prompt, image, depth_mask, mask,):
        size = (1024, 1024)
        depth_mask = self.preprocess_depth_mask(depth_mask, size)
        image = self.interpolate(image, size)
        image = self.normalize(image)
        mask = self.interpolate(mask, size)
        mask = self.normalize(mask)

        latent = self.sdxl_inpaint(prompt=prompt,
                                   num_inference_steps=20,
                                   image=image,
                                   mask_image=mask,
                                   control_image=depth_mask,
                                   controlnet_conditioning_scale=0.5,
                                   output_type="latent",
                                   ).images
        image = self.sdxl_refiner(
            prompt=prompt, image=latent, output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)

    def preprocess_depth_mask(self, depth_mask, size=(1024, 1024)):
        depth_mask = self.interpolate(depth_mask, size)
        depth_mask = self.normalize(depth_mask)
        depth_mask = torch.cat([depth_mask] * 3, dim=1)
        return depth_mask

    def interpolate(self, image, size=(1024, 1024)):
        return torch.nn.functional.interpolate(
            image, size=size, mode="bicubic", align_corners=False,
        )

    def normalize(self, image):
        min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
        max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
        return (image - min) / (max - min)
