from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionPipeline, DiffusionPipeline
from io import BytesIO
import base64
from auth_token import auth_token
from PIL import Image, ImageOps
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials = True,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

device = "cuda"
model = "timbrooks/instruct-pix2pix"
# pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, revision='fp16', torch_dtype=torch.float16, use_auth_token = auth_token)
pipeline = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", safety_checker=None, use_auth_token = auth_token)
# pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision='fp16', torch_dtype = torch.float16, use_auth_token = auth_token)
pipeline.to(device)
#pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)



def convert_to_pil_image(file: UploadFile) -> Image.Image:
    try:
        image_bytes = BytesIO(file.file.read())
        image = Image.open(image_bytes)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")

def download_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

@app.post("/")
# def generateImage(prompt: str, image: UploadFile = File(...)):
def generateImage():
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/250px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
    imagetest = download_image(url)
    prompttest = "add fireworks to the sky"
    # pil_image = convert_to_pil_image(image)
    with autocast(device):
        result = pipeline(prompttest, image=imagetest, num_inference_steps=10, image_guidance_scale=10).images[0]
        #result = pipeline(prompttest, guidance_scale=8.5).images[0]
    result.save("test.png")