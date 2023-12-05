from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
import torchvision
from typing import List
import numpy as np
from PIL import Image
from io import BytesIO
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open the config file using a context manager
with open("app/config.yaml") as f:
    config = yaml.safe_load(f)

# PRUEBA
imagen_Prueba = "1366_2000.jpg"
imagen = Image.open(imagen_Prueba)

# PRUEBA
with BytesIO() as output_bytes:
    imagen.save(output_bytes, format="JPEG")
    imagen_bytes = output_bytes.getvalue()

# Assign values from the dictionary to local variables
clases = config.get("class_names", [])
device = config.get("device", "cpu")
ruta_modelo = config.get("model_path", "mi_modelo.pth")
parametros_Transform = config.get("transform_params", {})

# PRUEBA
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(parametros_Transform.get("resize", 224), antialias=True),
    torchvision.transforms.CenterCrop(parametros_Transform.get("center_crop", 224)),
    torchvision.transforms.Normalize(parametros_Transform.get("mean", [0.485, 0.456, 0.406]),
                                     parametros_Transform.get("std", [0.229, 0.224, 0.225])),
])
image_tensor = transform(imagen).unsqueeze(0)

# Load the model
model = torchvision.models.vit_b_16(weights=None).to(device)
try:
    model.load_state_dict(torch.load(ruta_modelo, map_location=torch.device(device)), strict=False)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

# Set the model to evaluation mode
model.eval()

# Create a FastAPI instance
app = FastAPI()

# Define the response model
class PredictionResponse(BaseModel):
    class_id: str
    class_name: str
    probability: float

# Define the transform function
def transform_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Define the function to get predictions
def get_top_predictions(image_tensor, top_k=2):
    output = model(image_tensor)
    output = torch.softmax(output, dim=1)

    # Get the top k predictions along with their probabilities
    probabilities, indices = torch.topk(output, top_k, dim=1)

    # Convert indices to class names
    class_ids = [str(idx.item()) for idx in indices[0]]
    class_names = [clases[idx.item()] for idx in indices[0]]

    # Create a list of Pydantic model instances
    predictions = [
        PredictionResponse(class_id=id, class_name=name, probability=prob.item())
        for id, name, prob in zip(class_ids, class_names, probabilities[0])
    ]

    return predictions

# Define the predict endpoint
@app.post("/predict", response_model=List[PredictionResponse])
async def predict(file: UploadFile = File(...), top_k: int = 2):
    try:
        logger.info("Starting the predict function")

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="The file is empty")

        image = Image.open(BytesIO(image_bytes))

        if image.format.lower() not in ["jpeg", "jpg", "png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        image_tensor = transform_image(image_bytes=image_bytes)
        predictions = get_top_predictions(image_tensor=image_tensor, top_k=top_k)

        logger.info("Finished the predict function")

        return predictions
    except HTTPException as he:
        logger.error(f"HTTPException in predict function: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error in predict function: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Define the home endpoint
@app.get("/")
def home():
    return "Hello there"
