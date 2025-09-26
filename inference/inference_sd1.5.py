import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from PIL import Image


sd15_base = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet_path = "your_model"  


tokenizer = AutoTokenizer.from_pretrained(sd15_base, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_base, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_base, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(unet_path)
scheduler = DDIMScheduler.from_pretrained(sd15_base, subfolder="scheduler")



pipe = StableDiffusionInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
)
pipe.to("cuda", torch.float16)


input_path = "your_img_path"
img = Image.open(input_path).convert("RGB")
prompt = "generalize to scale 25"
generator = torch.Generator(device="cuda").manual_seed(42)

result = pipe(
    prompt,
    image=img,
    num_inference_steps=100,
    image_guidance_scale= 4.0,
    guidance_scale=4.0,
    generator=generator,
)
out_path = f"your_output_img_path"
result.images[0].save(out_path)
print(f"finish, save at {out_path}")
