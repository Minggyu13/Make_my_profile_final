import sys
from BLIP.models.blip_vqa import blip_vqa
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from BLIP.models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blip_vqa_model_path = 'model_base_vqa_capfilt_large.pth'
blip_model_path = 'model_base_capfilt_large.pth'


def BLIP_INFER(file_path):
    def load_image(image_size, device):
        img_path = file_path
        raw_image = Image.open(img_path).convert('RGB')
        #
        # w, h = raw_image.size
        # display(raw_image.resize((w // 5, h // 5)))

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        image = transform(raw_image).unsqueeze(0).to(device)
        return image


    image_size = 384
    image = load_image(image_size=image_size, device=device)

    model = blip_decoder(pretrained=blip_model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=True, num_beams=7, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])


    image_size = 480
    image = load_image(image_size=image_size, device=device)


    model = blip_vqa(pretrained=blip_vqa_model_path, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    question = 'what pose of hand'

    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')
        print('answer: ' + answer[0])


    return caption, answer
