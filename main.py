import base64
import io
from PIL import Image

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/object-detection', response_class=HTMLResponse)
async def object_detection_page(request: Request):
    return templates.TemplateResponse("object-detection.html", {"request": request})


@app.post("/object-detection/results")
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
    except Exception as e:
        return {"message": e}
    finally:
        file.file.close()

    # TODO: convert contents to a format that can be read with YOLOv8, write bounding boxes on image, and convert
    #  back to bytes (same type as the contents variable)
    base64_encoded_image = base64.b64encode(contents).decode("utf-8")

    return templates.TemplateResponse("object-detection.html", {"request": request, "img": base64_encoded_image})


@app.get('/object-tracking', response_class=HTMLResponse)
async def object_tracking_page(request: Request):
    return templates.TemplateResponse("object-tracking.html", {"request": request})


@app.get('/image-recognition', response_class=HTMLResponse)
async def image_recognition_page(request: Request):
    return templates.TemplateResponse("image-recognition.html", {"request": request})
