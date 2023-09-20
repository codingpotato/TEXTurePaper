import torch

from diffusers import AutoPipelineForInpainting, ControlNetModel, DDIMScheduler
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
from diffusers.utils import load_image
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

        self.sdxl_inpaint = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)

        self.sdxl_img2img = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            variant="fp16", use_safetensors=True, torch_dtype=torch.float16
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
            checker_mask = torch.nn.functional.interpolate(
                checker_mask, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            update_mask = torch.nn.functional.interpolate(
                update_mask, size=(1024, 1024), mode="bicubic",
                align_corners=False,
            )
            image = self.sdxl(prompt,
                              image=depth_image,
                              num_inference_steps=10,
                              controlnet_conditioning_scale=0.5,
                              output_type="np").images[0]
            image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image = self.sdxl_inpaint(prompt,
                                      num_inference_steps=20,
                                      image=image,
                                      mask_image=checker_mask,
                                      output_type="np",
                                      ).images[0]
            image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image = self.sdxl_inpaint(prompt,
                                      num_inference_steps=20,
                                      image=image,
                                      mask_image=update_mask,
                                      output_type="np",
                                      ).images[0]
            image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image = self.sdxl_img2img(prompt,
                                      image=image,
                                      control_image=depth_image,
                                      num_inference_steps=20,
                                      controlnet_conditioning_scale=0.5,
                                      output_type="np").images[0]
        else:
            image = self.sdxl(prompt,
                              image=depth_image,
                              num_inference_steps=50,
                              controlnet_conditioning_scale=0.5,
                              output_type="np").images[0]
        return transforms.ToTensor()(image).unsqueeze(0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdxl = SDXL(device)

    init_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
    )
    init_image = init_image.resize((1024, 1024))

    mask_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
    )
    mask_image = mask_image.resize((1024, 1024))

    image = sdxl("a handsome man with ray-ban sunglasses",
                 init_image, mask_image)
    image.save("experiments/sdxl.png")
