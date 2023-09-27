import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
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

        sd_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        ).to(device)

        self.sd_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=sd_controlnet,
            torch_dtype=torch.float16,
        ).to(device)

    def txt2img(self, prompt, depth_mask):
        depth_mask = self.preprocess_depth_mask(depth_mask)
        image = self.sdxl(prompt, image=depth_mask,
                          num_inference_steps=30,
                          guidance_scale=7.5,
                          controlnet_conditioning_scale=0.8,
                          output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)

    def img2img(self, prompt, image, depth_mask):
        depth_mask = self.preprocess_depth_mask(depth_mask)
        image = self.sdxl_img2img(prompt, image=image, control_image=depth_mask,
                                  num_inference_steps=30,
                                  guidance_scale=7.5,
                                  strength=0.8,
                                  controlnet_conditioning_scale=0.8,
                                  output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)

    def inpaint(self, prompt, image, mask, depth_mask):
        size = (512, 512)
        depth_512 = self.preprocess_depth_mask(depth_mask, size)
        image = self.interpolate(image, size)
        image = self.normalize(image)
        mask = self.interpolate(mask, size)
        mask = self.normalize(mask)

        image = self.sd_inpaint(prompt=prompt,
                                num_inference_steps=30,
                                image=image,
                                mask_image=mask,
                                control_image=depth_512,
                                guidance_scale=5,
                                strength=0.5,
                                output_type="np",
                                controlnet_conditioning_scale=0.8,
                                ).images[0]
        image = transforms.ToTensor()(image).unsqueeze(0)
        depth_1024 = self.preprocess_depth_mask(depth_mask, (1024, 1024))
        image = self.interpolate(image, (1024, 1024))
        image = self.normalize(image)
        image = self.sdxl_img2img(prompt,
                                  image=image,
                                  control_image=depth_1024,
                                  num_inference_steps=30,
                                  guidance_scale=7.5,
                                  strength=0.8,
                                  controlnet_conditioning_scale=0.8,
                                  output_type="np").images[0]
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
