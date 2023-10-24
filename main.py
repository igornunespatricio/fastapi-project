from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

import utils

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")

templates = Jinja2Templates(directory="templates")

model = None


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/object-detection', response_class=HTMLResponse)
def object_detection_page(request: Request):
    global model
    if model is None:
        model = YOLO('yolov8s.pt')
    return templates.TemplateResponse("object-detection.html", {"request": request})


@app.post("/object-detection/results", response_class=HTMLResponse)
def upload_image(request: Request, file: UploadFile = File(...)):
    global model
    if model is None:
        model = YOLO('yolov8s.pt')

    contents, results = utils.run_detect(fo=file.file, model=model)
    results = utils.results2df(results=results)
    file.file.close()
    return templates.TemplateResponse(
        "object-detection.html",
        {
            "request": request,
            "img": contents,
            "results": results
        }
    )


@app.get('/object-tracking', response_class=HTMLResponse)
def object_tracking_page(request: Request):
    return templates.TemplateResponse("object-tracking.html", {"request": request})


@app.get('/image-recognition', response_class=HTMLResponse)
def image_recognition_page(request: Request):
    return templates.TemplateResponse("image-recognition.html", {"request": request})
