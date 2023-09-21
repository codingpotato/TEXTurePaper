import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetInpaintPipeline
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
            depth_image = torch.nn.functional.interpolate(
                depth_image, size=(512, 512), mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_image, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_image, dim=[1, 2, 3], keepdim=True)
            depth_image = (depth_image - depth_min) / (depth_max - depth_min)
            depth_image = torch.cat([depth_image] * 3, dim=1)

            image = torch.nn.functional.interpolate(
                image, size=(512, 512), mode="bicubic",
                align_corners=False,
            )
            min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
            max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
            image = (image - min) / (max - min)

            checker_mask = torch.nn.functional.interpolate(
                checker_mask, size=(512, 512), mode="bicubic",
                align_corners=False,
            )
            checker_min = torch.amin(checker_mask, dim=[1, 2, 3], keepdim=True)
            checker_max = torch.amax(checker_mask, dim=[1, 2, 3], keepdim=True)
            checker_mask = (checker_mask - checker_min) / \
                (checker_max - checker_min)

            update_mask = torch.nn.functional.interpolate(
                update_mask, size=(512, 512), mode="bicubic",
                align_corners=False,
            )
            update_min = torch.amin(update_mask, dim=[1, 2, 3], keepdim=True)
            update_max = torch.amax(update_mask, dim=[1, 2, 3], keepdim=True)
            update_mask = (update_mask - update_min) / \
                (update_max - update_min)

            image = self.sd_inpaint(prompt=prompt,
                                    num_inference_steps=50,
                                    image=image,
                                    mask_image=update_mask,
                                    control_image=depth_image,
                                    guidance_scale=8.0,
                                    strength=0.99,
                                    controlnet_conditioning_scale=0.8,
                                    output_type="np",
                                    ).images[0]
        else:
            depth_image = torch.nn.functional.interpolate(
                depth_image, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_image, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_image, dim=[1, 2, 3], keepdim=True)
            depth_image = (depth_image - depth_min) / (depth_max - depth_min)
            depth_image = torch.cat([depth_image] * 3, dim=1)

            image = self.sdxl(prompt,
                              image=depth_image,
                              num_inference_steps=50,
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
