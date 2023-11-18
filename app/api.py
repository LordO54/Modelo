print("hello world")

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torchvision
import numpy as np
from PIL import Image
from io import BytesIO

class_names= ['sopa de lentejas',
  'sopa de lentejas - arroz',
  'sopa de lentejas - con chorizo',
  'sopa de lentejas - con pan']

device ="cpu"

# Crea una instancia del mismo tipo de modelo que usaste para entrenar
model = torchvision.models.vit_b_16(weights=None).to(device)

# Carga el estado del modelo desde el archivo 'mi_modelo.pth'
model.load_state_dict(torch.load('mi_modelo.pth', map_location=torch.device('cpu')),strict=False)


# Crea una instancia de FastAPI
app = FastAPI()

# Define la clase para el esquema de la respuesta de la API
class Prediction(BaseModel):
    class_id: str
    class_name: str

# Define la función para transformar la imagen en un tensor compatible con el modelo
def transform_image(image_bytes):
    # Usa las mismas transformaciones que se usaron para entrenar el modelo
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    # Convierte la imagen en un tensor
    image = Image.open(image_bytes)
    return transform(image).unsqueeze(0)

# Define la función para cargar el modelo desde el archivo y hacer la predicción sobre la imagen
def get_prediction(image_tensor):
    # Carga el modelo preentrenado de ViT-B-16
    model = torchvision.models.vit_b_16(weights=None).to("cpu")
    # Carga el estado del modelo desde el archivo 'mi_modelo.pth'
    model.load_state_dict(torch.load('mi_modelo.pth', map_location=torch.device('cpu')),strict=False)
    # Pone el modelo en modo de evaluación
    model.eval()
    # Obtiene la salida del modelo sobre el tensor de imagen
    output = model(image_tensor)
    # Obtiene la clase predicha y su nombre
    _, index = torch.max(output, 1)
    class_id = str(index.item())
    class_name = class_names[index]
    return class_id, class_name

# Define la ruta de la API para recibir una imagen y devolver una predicción
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Lee el contenido del archivo subido
    image_bytes = await file.read()
    # Convierte los bytes en un objeto BytesIO
    image_io = BytesIO(image_bytes)
    # Convierte el objeto BytesIO en un objeto Image
    image = Image.open(image_io)
    # Transforma la imagen en un tensor
    image_tensor = transform_image(image_io)
    # Obtiene la predicción del modelo sobre el tensor de imagen
    class_id, class_name = get_prediction(image_tensor=image_tensor)
    # Devuelve la respuesta de la API con la clase predicha y su nombre
    return {"class_id": class_id, "class_name": class_name}

@app.get("/")
def home():
    return "hello there"