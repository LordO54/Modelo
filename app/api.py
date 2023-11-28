print("hello world")

from fastapi import FastAPI, File, UploadFile,HTTPException
from pydantic import BaseModel
import torch
import torchvision
import numpy as np
from PIL import Image
from io import BytesIO
import yaml

#abre el archivo config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

#cierra el archivo
f.close

#asigna los valores del diccionario a variables locales
clases=config["class_names"]
device=config["device"]
ruta_modelo=config["model_path"]
parametros_Transform=config["transform_params"]

# Crea una instancia del mismo tipo de modelo que usaste para entrenar
model = torchvision.models.vit_b_16(weights=None).to(device)

# Carga el estado del modelo desde el archivo 'mi_modelo.pth'
model.load_state_dict(torch.load('mi_modelo.pth', map_location=torch.device('cpu')),strict=False)

#poner el modelo en modo evaluacion
model.eval()


# Crea una instancia de FastAPI
app = FastAPI()

# Define la clase para el esquema de la respuesta de la API
class Prediction(BaseModel):
    predictions: list[tuple[str, str]]



# Define la función para transformar la imagen en un tensor compatible con el modelo
def transform_image(image_bytes):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(parametros_Transform["resize"]),
        torchvision.transforms.CenterCrop(parametros_Transform["center_crop"]),
        torchvision.transforms.Normalize(parametros_Transform["mean"],parametros_Transform["std"]),
        torchvision.transforms.ToTensor(),
    ])
    #convierte la imagen en tensor
    image=Image.open(image_bytes)
    return transform(image).unsqueeze(0)

# Define la función para cargar el modelo desde el archivo y hacer la predicción sobre la imagen
def get_prediction(image_tensor):
    #Salida del modelo en base a la imagen
    output = model(image_tensor)

    #Aplica la funcion softmax para obtener la prediccion
    output=torch.softmax(output,dim=1)

    #obtiene los k valores mas altos y sus indicies
    values, indicies= torch.topk(output,k=3)

    #Se crea una lista vacia de predicciones
    preddiciones=[]

    #Recorre ls indices y valores de 
    for i, v in zip(indicies[0],values[0]):
        #obtiene el valor correspodiente a la clase
        class_name=clases[i]

        #convierte el numero de la probabilidad en un numero decimal
        probabilidad=round(v.item(),2)

        #se anade la tupla a la lista de predicciones

    
    #devulve la lista de tuplas con las 3 respuestas mas probables
    return preddiciones

    # Obtiene la clase predicha y su nombre
    _, index = torch.max(output, 1)
    class_id = str(index.item())
    class_name = class_names[index]
    return class_id, class_name

# Define la ruta de la API para recibir una imagen y devolver una predicción
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):

    try:

        # Lee el contenido del archivo subido
        image_bytes = await file.read()

        #valida que el archivo no este vacio
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo esta vacio")
        
        
        # Convierte los bytes en un objeto BytesIO
        image_io = BytesIO(image_bytes)

        # Convierte el objeto BytesIO en un objeto Image
        image = Image.open(image_io)

        #Se valida que el archivo tenga un formato correcto
        if image.format not in ["JPG", "PNG"]:
            raise HTTPException(status_code=400, detail="El formato no es valido")
        

        # Transforma la imagen en un tensor, una vez aplicadas las trasnformaciones predeterminadas
        image_tensor = transform_image(image_io)


        # Obtiene la predicción del modelo sobre el tensor de imagen
        Prediction = get_prediction(image_tensor=image_tensor)
        # Devuelve la respuesta de la API con la clase predicha y su nombre
        return {"Predictions": Prediction}
    except Exception as e:
        #control de cualquier otro error que pudiera ocurrir
        HTTPException(status_code=500, detail=str(e))
        

@app.get("/")
def home():
    return "hello there"