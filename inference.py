from diffusers import StableDiffusionPipeline
import torch
import datetime
from arg_parser import get_args


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    repo_id = args.repo_id or "DJMOON/train_lora"
    prompt = args.prompt
    
    pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")

    # safety checker disabled
    def return_as_it_is(images, **kwargs):
        return images, False
    pipeline.safety_checker = return_as_it_is
    
    image = pipeline(prompt).images[0]
    image.save(f"inference_{timestamp}.png")
    print("...inference finished")


if __name__ == "__main__":
    args = get_args()
    main(args)
