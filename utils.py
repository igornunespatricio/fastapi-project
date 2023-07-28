import io
import base64

from PIL import Image
from ultralytics import YOLO
import numpy as np

def run_detect(fo, model: YOLO):
    im = Image.open(fo)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    results = model.predict(source=im)
    # TODO: return id, confidence and boxes from results 
    im = results[0].plot(conf=True, line_width=2) # image as numpy ndarray
    im = im[:,:,::-1] # converting BGR to RGB
    im = Image.fromarray(im) # reading image as Pil
    with io.BytesIO() as buf:
        im.save(buf, format='JPEG')
        contents = buf.getvalue()
    contents = base64.b64encode(contents).decode("utf-8")
    im.close()
    return contents
