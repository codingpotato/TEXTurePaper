import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionXLControlNetInpaintPipeline
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

        self.sdxl_inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=controlnet,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

    def __call__(self, use_inpaint, prompt, image, depth_image, update_mask,
                 checker_mask):
        depth_image = torch.nn.functional.interpolate(
            depth_image, size=(1024, 1024), mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_image, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_image, dim=[1, 2, 3], keepdim=True)
        depth_image = (depth_image - depth_min) / (depth_max - depth_min)
        depth_image = torch.cat([depth_image] * 3, dim=1)

        if use_inpaint:
            image = torch.nn.functional.interpolate(
                image, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
            max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
            image = (image - min) / (max - min)

            checker_mask = torch.nn.functional.interpolate(
                checker_mask, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            update_mask = torch.nn.functional.interpolate(
                update_mask, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            update_min = torch.amin(update_mask, dim=[1, 2, 3], keepdim=True)
            update_max = torch.amax(update_mask, dim=[1, 2, 3], keepdim=True)
            update_mask = (update_mask - update_min) / \
                (update_max - update_min)

            image = self.sdxl_inpaint(prompt,
                                      num_inference_steps=20,
                                      image=image,
                                      control_image=depth_image,
                                      mask_image=update_mask,
                                      guidance_scale=8.0,
                                      strength=0.99,
                                      controlnet_conditioning_scale=0.8,
                                      output_type="np",
                                      ).images[0]
        else:
            image = self.sdxl(prompt,
                              image=depth_image,
                              num_inference_steps=20,
                              guidance_scale=5.0,
                              controlnet_conditioning_scale=0.8,
                              output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)


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
