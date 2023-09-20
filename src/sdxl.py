import torch

from diffusers import AutoPipelineForInpainting, ControlNetModel, DDIMScheduler
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image


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
        self.sdxl.enable_model_cpu_offload()

        self.sdxl_inpaint = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            variant="fp16", torch_dtype=torch.float16,
        ).to(device)
        self.sdxl_inpaint.enable_model_cpu_offload()

    def __call__(self, prompt, image, mask_image):
        return self.sdxl_inpaint(prompt,
                                 num_inference_steps=20,
                                 eta=1.0,
                                 image=image,
                                 mask_image=mask_image,
                                 output_type="pil"
                                 ).images[0]


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
