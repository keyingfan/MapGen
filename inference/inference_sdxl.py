import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline, PNDMScheduler,UNet2DConditionModel,DPMSolverMultistepScheduler, AutoencoderKL, EulerAncestralDiscreteScheduler
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from PIL import Image
from diffusers import EulerDiscreteScheduler,DDIMScheduler,UniPCMultistepScheduler


sdxl_base = "stabilityai/stable-diffusion-xl-base-1.0"
unet_path = "your_model"
vae_path = sdxl_base  


tokenizer1 = AutoTokenizer.from_pretrained(sdxl_base, subfolder="tokenizer")
tokenizer2 = AutoTokenizer.from_pretrained(sdxl_base, subfolder="tokenizer_2")
text_encoder1 = CLIPTextModel.from_pretrained(sdxl_base, subfolder="text_encoder")
text_encoder2 = CLIPTextModelWithProjection.from_pretrained(sdxl_base, subfolder="text_encoder_2")
vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet")
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(sdxl_base, subfolder="scheduler")


pipe = StableDiffusionXLInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder1,
    text_encoder_2=text_encoder2,
    tokenizer=tokenizer1,
    tokenizer_2=tokenizer2,
    unet=unet,
    scheduler=scheduler,
)
pipe.to("cuda",torch.float16)


input_path = "your_img_path"
img = Image.open(input_path).convert("RGB")
prompt = "generalize to scale 15"


generator = torch.Generator(device="cuda").manual_seed(30)


steps_list = [1,50,75,100]
for steps in steps_list:
    result = pipe(
        prompt,
        image=img,
        num_inference_steps=steps,
        image_guidance_scale=3.0,
        guidance_scale=3.0,
        width=1024,
        height=1024,
        generator=generator,   
    )
    out_path = f"your_output_img_path"
    result.images[0].save(out_path)
    print(f"finish, save at {out_path}")
