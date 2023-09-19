import math
import numpy as np
import time
import torch

from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess
from ldm.util import instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms


class Zero123:
    def __init__(self, device):
        self.device = device
        config = OmegaConf.load(
            'configs/sd-objaverse-finetune-c_concat-256.yaml')
        ckpt = 'pretrained/zero123/zero123-xl.ckpt'
        self.zero123 = self.load_model_from_config(config, ckpt)
        self.carvekit = create_carvekit_interface()

    def load_model_from_config(self, config, ckpt, verbose=False):
        pl_sd = torch.load(ckpt, map_location='cpu')

        if 'global_step' in pl_sd and verbose:
            print(f'[INFO] Global Step: {pl_sd["global_step"]}')

        sd = pl_sd['state_dict']

        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print('[INFO] missing keys: \n', m)
        if len(u) > 0 and verbose:
            print('[INFO] unexpected keys: \n', u)

        torch.cuda.empty_cache()
        model.eval().to(device)
        return model

    @torch.no_grad()
    def sample_model(self, input_img, sampler, precision, h, w, ddim_steps,
                     scale, ddim_eta, x, y, z):
        precision_scope = autocast if precision == 'autocast' else nullcontext
        with precision_scope(self.device):
            with self.zero123.ema_scope():
                c = self.zero123.get_learned_conditioning(input_img)
                c = c.tile(1, 1, 1)
                T = torch.tensor([math.radians(x),
                                  math.sin(math.radians(y)),
                                  math.cos(math.radians(y)),
                                  z])
                T = T[None, None, :].repeat(1, 1, 1).to(c.device)
                c = torch.cat([c, T], dim=-1)
                c = self.zero123.cc_projection(c)
                cond = {}
                cond['c_crossattn'] = [c]
                cond['c_concat'] = [self.zero123.encode_first_stage(
                    (input_img.to(c.device))).mode().detach().repeat(1, 1, 1, 1)]
                if scale != 1.0:
                    uc = {}
                    uc['c_concat'] = [torch.zeros(
                        1, 4, h // 8, w // 8).to(c.device)]
                    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                else:
                    uc = None

                shape = [4, h // 8, w // 8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=None)
                print(samples_ddim.shape)
                # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
                x_samples_ddim = self.zero123.decode_first_stage(samples_ddim)
                return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

    def __call__(self, x=0.0, y=0.0, z=0.0, raw_img=None, scale=3.0,
                 ddim_steps=50, ddim_eta=1.0, precision='fp16', h=256, w=256):
        '''
        :param raw_img (PIL Image).
        '''
        raw_img.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
        input_img = self.preprocess_image(raw_img)

        input_img = transforms.ToTensor()(input_img).unsqueeze(0).to(self.device)
        input_img = input_img * 2 - 1
        input_img = transforms.functional.resize(input_img, [h, w])

        sampler = DDIMSampler(self.zero123)
        x_samples_ddim = self.sample_model(
            input_img, sampler, precision, h, w, ddim_steps, scale, ddim_eta,
            x, y, z)

        image = 255.0 * \
            rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')
        return Image.fromarray(image.astype(np.uint8))

    def preprocess_image(self, input_img, preprocess):
        '''
        :param input_img (PIL Image).
        :return input_img (H, W, 3) array in [0, 1].
        '''

        print('old input_img:', input_img.size)
        start_time = time.time()

        if preprocess:
            input_img = load_and_preprocess(self.carvekit, input_img)
            input_img = (input_img / 255.0).astype(np.float32)
            # (H, W, 3) array in [0, 1].
        else:
            input_img = input_img.resize([256, 256], Image.Resampling.LANCZOS)
            input_img = np.asarray(input_img, dtype=np.float32) / 255.0
            # (H, W, 4) array in [0, 1].

            # old method: thresholding background, very important
            # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

            # new method: apply correct method of compositing to avoid sudden
            # transitions / thresholding (smoothly transition foreground to
            # white background based on alpha values)
            alpha = input_img[:, :, 3:4]
            white_im = np.ones_like(input_img)
            input_img = alpha * input_img + (1.0 - alpha) * white_im

            input_img = input_img[:, :, 0:3]
            # (H, W, 3) array in [0, 1].

        print(
            f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
        print('new input_im:', lo(input_img))

        return input_img


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zero123 = Zero123(device)
    raw_img = Image.open('../images/hulk.png')
    image = zero123(raw_img=raw_img, y=45.0)
    image.save('../images/output.png')
