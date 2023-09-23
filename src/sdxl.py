import torch

from diffusers import ControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline


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
        self.pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)

    def __call__(self, image, depth_mask, prompt, negative_prompt=None):
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

        if image is not None:
            image = torch.nn.functional.interpolate(
                image, size=(1024, 1024), mode="bicubic", align_corners=False,
            )
            image_min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
            image_max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
            image = (image - image_min) / (image_max - image_min)

            images = self.pipe_refiner(prompt=prompt, image=image,
                                       negative_prompt=negative_prompt,
                                       num_inference_steps=30,
                                       output_type="np").images
        else:
            images = self.pipe(prompt=prompt, image=depth_mask,
                               negative_prompt=negative_prompt,
                               num_inference_steps=30,
                               guidance_scale=0.75,
                               controlnet_conditioning_scale=0.8,
                               output_type="np").images

        return images[0]

    def txt2img(self, prompt, depth_mask, negative_prompt=None):
        latent = self.pipe(prompt=prompt,
                           image=self.preprocess_depth(depth_mask),
                           negative_prompt=negative_prompt,
                           num_inference_steps=30,
                           output_type="latent").images
        return self.pipe_refiner(prompt=prompt, image=latent,
                                 negative_prompt=negative_prompt,
                                 num_inference_steps=30,
                                 output_type="np").images[0]

    def preprocess_depth(self, depth_mask):
        depth_mask = self.normalize(depth_mask)
        depth_mask = self.interpolate(depth_mask, (1024, 1024))
        return torch.cat([depth_mask] * 3, dim=1)

    def interpolate(self, image, size):
        return torch.nn.functional.interpolate(
            image, size, mode="bicubic", align_corners=False,
        )

    def normalize(self, image):
        min = torch.amin(image, dim=[1, 2, 3], keepdim=True)
        max = torch.amax(image, dim=[1, 2, 3], keepdim=True)
        return (image - min) / (max - min)


if __name__ == "__main__":
    import argparse
    import torch
    from torchvision.io import read_image
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a painting of a bird",
                        help="prompt")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="negative prompt")
    parser.add_argument("--depth_mask", type=str, default=None,
                        help="depth mask")
    parser.add_argument("--output", type=str, default="output.png",
                        help="output image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdxl = SDXL(device)

    prompt = args.prompt
    negative_prompt = args.negative_prompt
    depth_mask = args.depth_mask
    output = args.output

    if depth_mask is not None:
        depth_mask = read_image(depth_mask).unsqueeze(0)

    image = sdxl.txt2img(prompt, depth_mask, negative_prompt)
    save_image(torch.from_numpy(image).permute(2, 0, 1), output)
