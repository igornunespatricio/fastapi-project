import io
import base64

import pandas as pd
from PIL import Image
from ultralytics import YOLO
import numpy as np


def run_detect(fo, model: YOLO):
    im = Image.open(fo)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    results = model.predict(source=im)
    im = results[0].plot(conf=True, line_width=2)  # image as numpy ndarray
    im = im[:, :, ::-1]  # converting BGR to RGB
    im = Image.fromarray(im)  # reading image as Pil
    with io.BytesIO() as buf:
        im.save(buf, format='JPEG')
        contents = buf.getvalue()
    contents = base64.b64encode(contents).decode("utf-8")
    im.close()
    return contents, results


def results2df(results):
    results_df = pd.DataFrame()
    for result in results:
        names = result.names
        cls = [names.get(int(item)) for item in result.boxes.cls]
        xyxy = result.boxes.xyxy
        x0 = xyxy[:, 0]
        x1 = xyxy[:, 1]
        y0 = xyxy[:, 2]
        y1 = xyxy[:, 3]
        data = {'Object': cls, 'Confidence': result.boxes.conf,
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
        temp_results = pd.DataFrame(data)
        results_df = pd.concat([results_df, temp_results])
    return results_df.to_html(justify='center', index=False)
