import pandas as pd
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.predict(
    source=[
        r"C:\Users\IgorNunes\Downloads\star wars 2.jpg",
        r"C:\Users\IgorNunes\Downloads\star wars.jpg"
    ]
)

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

print(results_df.to_html())
