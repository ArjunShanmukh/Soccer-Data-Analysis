

from ultralytics import YOLO

model = YOLO('models/bestyolov5n.pt')

results = model.predict('football_analytics_project.avi', save=True)
print(results[0])
print('============================')
for box in results[0].boxes:
  print(box)