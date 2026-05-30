# Importo las librerias necesarias
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pickle
import numpy as np

# Cargamos nuestros modelos
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargamos el archivo para estandarizar
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Creamos la API
app = FastAPI()

# Crear enlace a la carpeta static o montamos la carpeta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Creamos el template que se va a ejecutar cuando levantemos el servicio y entramos al backend
templates = Jinja2Templates(directory="templates")


# Creo el endpoint para el consumo de la API
# Formulario enlace
@app.get("/", response_class=HTMLResponse)
# Creamos una función para mostrar el index
def formulario(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": None, "mensaje": None},
    )


# Implementamos o creamos una función para la predicción
@app.post("/predict", response_class=HTMLResponse)

# Creo la función para obtener los parámetros del formulario
def predecir(
    request: Request,
    atributo1: float = Form(...),
    atributo2: float = Form(...),
    atributo3: float = Form(...),
    atributo4: float = Form(...),
    atributo5: float = Form(...),
    atributo6: float = Form(...),
    atributo7: float = Form(...),
    atributo8: float = Form(...),
    atributo9: float = Form(...),
    atributo10: float = Form(...),
    atributo11: float = Form(...),
    atributo12: float = Form(...),
    atributo13: float = Form(...),
):
    # convertimos los datos a un arreglo dimensional
    datos = np.array(
        [
            [
                atributo1,
                atributo2,
                atributo3,
                atributo4,
                atributo5,
                atributo6,
                atributo7,
                atributo8,
                atributo9,
                atributo10,
                atributo11,
                atributo12,
                atributo13,
            ]
        ]
    )
    # Escalamos los nuevos datos
    datos_estandarizados = scaler.transform(datos)
    # Realizamos la predicción utilizando el modelo de aprendizaje
    prediccion = model.predict(datos_estandarizados)
    # Mostramos el resultado de la predicción
    resultado = int(prediccion[0])
    # Mostramos los mensajes indicados
    mensaje = ""
    # Hago condicion para verificar el resultado de la predicción
    if resultado == 0:
        mensaje = (
            "El paciente se encuentra SANO (Ausencia de enfermedad cardíaca)"
        )
    elif resultado == 1:
        mensaje = "El paciente se encuentra ENFERMO (Presencia de enfermedad cardíaca)"
    # Retornamos los resultados
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": resultado, "mensaje": mensaje},
    )
