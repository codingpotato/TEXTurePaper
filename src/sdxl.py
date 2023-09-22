import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLControlNetPipeline
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

        self.sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

        controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16
        ).to(device)

        def disabled_safety_checker(images, clip_input):
            return images, [False]

        self.sd_inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet_depth,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)
        self.sd_inpaint.safety_checker = disabled_safety_checker

    def __call__(self, use_inpaint, prompt, image, depth_image, update_mask,
                 checker_mask):
        if use_inpaint:
            depth_image = self.interpolate(depth_image, size=(512, 512))
            depth_image = self.normalize(depth_image)
            depth_image = torch.cat([depth_image] * 3, dim=1)

            image = self.interpolate(image, size=(512, 512))
            image = self.normalize(image)

            checker_mask = self.interpolate(checker_mask, size=(512, 512))
            checker_mask = self.normalize(checker_mask)

            update_mask = self.interpolate(update_mask, size=(512, 512))
            update_mask = self.normalize(update_mask)

            image = self.sd_inpaint(prompt=prompt,
                                    num_inference_steps=10,
                                    image=image,
                                    mask_image=checker_mask,
                                    control_image=depth_image,
                                    guidance_scale=8.0,
                                    strength=0.99,
                                    controlnet_conditioning_scale=0.8,
                                    output_type="np",
                                    ).images[0]
            image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image = self.interpolate(image, size=(1024, 1024))
            image = self.normalize(image)
            image = self.sdxl_refiner(prompt=prompt, image=image,
                                      output_type="np").images[0]
        else:
            depth_image = self.interpolate(depth_image, size=(1024, 1024))
            depth_image = self.normalize(depth_image)
            depth_image = torch.cat([depth_image] * 3, dim=1)
            image = self.sdxl(prompt,
                              image=depth_image,
                              num_inference_steps=50,
                              controlnet_conditioning_scale=0.8,
                              output_type="np").images[0]

        return transforms.ToTensor()(image).unsqueeze(0)

    def interpolate(self, image, size):
        return torch.nn.functional.interpolate(
            image, size=size, mode="bicubic", align_corners=False,
        )

    def normalize(self, image):
        min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
        max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
        return (image - min) / (max - min)


if __name__ == "__main__":
    from torchvision.utils import load_image, save_image
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-hybrid-midas")

    def get_depth_map(image):
        image = feature_extractor(
            image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad(), torch.autocast(device):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        return depth_map

    sdxl = SDXL(device)

    image = load_image(
        'https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png'
    )
    image = image.resize((1024, 1024))

    depth_image = get_depth_map(image)

    for i in range(5):
        image = sdxl(use_inpaint=False,
                     prompt="hulk, marvel movie character, realistic, high detailed, 8k",
                     image=None,
                     depth_image=depth_image,
                     update_mask=None,
                     checker_mask=None)
        save_image(image, f"experiments/sdxl_{i}.png")
