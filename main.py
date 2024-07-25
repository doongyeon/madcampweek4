from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from network.BL_network import BL_network
import requests
import base64

app = FastAPI()

# OpenAI API key
api_key = ""

class SimpleImageLoader:
    def __init__(self, image_path, patch_size=224):
        self.image_path = image_path
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            self.Crop_patches(self.patch_size)
        ])

    def load_image(self):
        image = Image.open(self.image_path).convert('RGB')
        image = self.transform(image)
        return image

    class Crop_patches:
        def __init__(self, size):
            self.size = size if isinstance(size, tuple) else (size, size)

        def __call__(self, img):
            img = img.permute(1, 2, 0).numpy()  # Convert CHW to HWC format
            patches = []
            stride = self.size[0] // 2
            h, w, _ = img.shape  # Get height and width from shape

            if h < self.size[0] or w < self.size[1]:
                raise ValueError(f"Image size ({h}, {w}) is smaller than patch size {self.size}")

            for i in range(0, h - self.size[0] + 1, stride):
                for j in range(0, w - self.size[1] + 1, stride):
                    patch = img[i:i+self.size[0], j:j+self.size[1], :]
                    patches.append(patch)

            if not patches:
                raise ValueError("No patches were created, check the image size and patch size/stride.")

            return torch.stack([transforms.ToTensor()(p) for p in patches])

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = BL_network().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    best_srcc = checkpoint['best_srcc']
    best_plcc = checkpoint['best_plcc']
    
    print(f'Loaded model from epoch {epoch} with best SRCC {best_srcc} and best PLCC {best_plcc}')
    
    return model, optimizer, epoch, best_srcc, best_plcc

def predict_image_score(model, image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        image = image.cuda(non_blocking=True)
        pred = model(image)
        score = np.mean(pred.cpu().numpy())
    return float(score)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Initialize the model once to avoid reloading for every request
model, optimizer, epoch, best_srcc, best_plcc = load_checkpoint('best_model_BL_res.pth')

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"uploaded_images/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Load and process the image for scoring
    try:
        image_loader = SimpleImageLoader(file_path)
        image = image_loader.load_image()
        score = predict_image_score(model, image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    return JSONResponse(content={"filename": file.filename, "score": score})

@app.post("/image-feedback/")
async def image_feedback(filename: str = Form(...)):
    file_path = f"uploaded_images/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Load the image and predict the score
    try:
        image_loader = SimpleImageLoader(file_path)
        image = image_loader.load_image()
        score = predict_image_score(model, image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    # Encode the image in base64 format
    try:
        base64_image = encode_image(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Include the score in the GPT request message
    # message_content = f"이 사진의 점수는 100점 만점 중 {score}이야. 조금 더 높은 점수를 얻기 위해서는 어떻게 사진을 촬영하는 것이 좋을까? 조명, 구도, 색감,포커스의 요소들에 대해서 피드백 해줘. 만약 이 사진이 구체적인 피사체가 없는 사진이라면 피사체를 명확하게 해달라는 응답을 보내줘"
    message_content = f"이 사진의 점수는 100점 만점 중 {score}이야. 이 사진에 대해 설명해주고 더 너안 사진을 찍은 위한 방법들에 대해 알려줘"
    # message_content = "이 사진에 대해 0점부터 100점까지 점수를 매겨주고 그 이유에 대해 설명해줘 그리고 더 개선할 방안이 있는지 알려줘. 점수는 어떤 경우에서도 무조건 응답해야해. 응답은 한국어로 해"
    print(message_content)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        feedback = response.json()
        print(feedback)
        content = feedback['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail="Unexpected response format from OpenAI API.")

    return JSONResponse(content={"filename": filename, "score": score, "feedback": content})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
