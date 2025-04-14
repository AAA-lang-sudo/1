import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, PretrainedConfig

import torch
import torch.nn.functional as F
from torchvision import transforms

import diffusers
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,

)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.utils import check_min_version
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin

check_min_version("0.17.0")

def get_model_class(model_path, rev):
    encoder_config = PretrainedConfig.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=rev,
    )
    class_name = encoder_config.architectures[0]

    if class_name == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel

    elif class_name == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel

    elif class_name == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation

    else:
        raise ValueError(f"{class_name} 不被支持。")


def process_prompt(tokenizer, input_prompt, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length

    text_input = tokenizer(
        input_prompt,
        max_length=max_len,
        return_tensors="pt",
        truncation=True,
        padding="max_length",

    )

    return text_input


def get_prompt_embedding(encoder, input_ids, attention_mask, use_attention_mask=False):
    input_ids = input_ids.to(encoder.device)

    if use_attention_mask:
        attention_mask = attention_mask.to(encoder.device)
    else:
        attention_mask = None

    embeds = encoder(
        input_ids,
        attention_mask=attention_mask,
    )
    embeds = embeds[0]

    return embeds


def lora_training(img,
                  prompt_str,
                  model_dir,
                  batch_size,
                  rank,
                  vae_dir,
                  save_path,
                  steps,
                  learning_rate,
                  prog,
                  interval=-1):
    set_seed(0)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    noise_sched = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    text_encoder_cls = get_model_class(model_dir, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_dir, subfolder="text_encoder", revision=None
    )
    if vae_dir == "default":
        vae = AutoencoderKL.from_pretrained(
            model_dir, subfolder="vae", revision=None
        )
    else:
        vae = AutoencoderKL.from_pretrained(vae_dir)
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", revision=None
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    unet_lora_attn = {}

    for name, attn_proc in unet.attn_processors.items():
        cross_attn_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

        else:
            raise NotImplementedError("改名称头")

        if isinstance(attn_proc, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn[name] = lora_attn_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attn_dim, rank=rank
        )

    unet.set_attn_processor(unet_lora_attn)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    params = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_sched = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=steps,
        num_cycles=1,
        power=1.0,
    )

    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_sched = accelerator.prepare_scheduler(lr_sched)

    with torch.no_grad():
        text_input = process_prompt(tokenizer, prompt_str, tokenizer_max_length=None)
        text_embeds = get_prompt_embedding(
            text_encoder,
            text_input.input_ids,
            text_input.attention_mask,
            text_encoder_use_attention_mask=False
        )
        text_embeds = text_embeds.repeat(batch_size, 1, 1)

    img_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    for step in tqdm(range(steps), desc="调整"):
        unet.train()
        img_batch = []
        for _ in range(batch_size):
            transformed_img = img_transforms(Image.fromarray(img)).to(device, dtype=torch.float16)
            transformed_img = transformed_img.unsqueeze(dim=0)
            img_batch.append(transformed_img)

        img_batch = torch.cat(img_batch, dim=0)
        latents_dist = vae.encode(img_batch).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape

        timesteps = torch.randint(
            0, noise_sched.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        noisy_input = noise_sched.add_noise(model_input, noise, timesteps)

        model_pred = unet(noisy_input, timesteps, text_embeds).sample

        if noise_sched.config.prediction_type == "epsilon":
            target = noise
        elif noise_sched.config.prediction_type == "v_prediction":
            target = noise_sched.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"未知类型 {noise_sched.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad()

        if interval > 0 and (step + 1) % interval == 0:
            intermediate_path = os.path.join(save_path, str(step + 1))
            if not os.path.isdir(intermediate_path):
                os.mkdir(intermediate_path)

            LoraLoaderMixin.save_lora_weights(
                save_directory=intermediate_path,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )

    LoraLoaderMixin.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )
