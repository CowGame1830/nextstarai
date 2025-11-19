from ultralytics import YOLO 

model = YOLO('models/clean_label.pt')

results = model.predict('input_videos/10secVideo.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)