import io

from PIL import Image
from ultralytics import YOLO


def run_detect(fo, model: YOLO):
    im = Image.open(fo)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    results = model.predict(source=im)
    # TODO: get the image in the correct format to contents
    im = results[0].plot(pil=True, conf=True, line_width=2)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=50)
    contents = buf.getvalue()  # get content as bytes
    # buf.seek(0)  # return the stream position to the start
    buf.close()
    im.close()
    return contents
