from ultralytics import YOLO 

model = YOLO('models/model_2_0.pt')  

results = model.predict('input_videos/1.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)