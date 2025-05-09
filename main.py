from project_file import *
from pathlib import Path

data_path = Path('C:/Users/msen6/Documents/Github Projects/datasets/flower-dataset')
model_fe = "./models/model_feature_extraction.pth"
model_sc = "./models/model_scratch.pth"
model_ft = "./models/model_fine_tuning.pth"
models = [model_fe, model_sc, model_ft]
results = {}
image_2 = "./test/Image_2.jpg"
image_5 = "./test/Image_5.jpg"
image_6 = "./test/Image_6.jpg"
image_19 = "./test/Image_19.jpg"
image_34 = "./test/Image_34.jpg"
images = [image_2, image_5, image_6, image_19, image_34]


for model_path in models:
    print(f"\n\n==== Loading {model_path.upper()} Model ====")
    # Determine model type from the path
    if "feature_extraction" in model_path:
        model_type = "feature_extraction"
    elif "scratch" in model_path:
        model_type = "scratch"
    elif "fine_tuning" in model_path:
        model_type = "fine_tuning"
    else:
        print(f"Unknown model type for {model_path}")
        continue
        
    model = Custom_Model(model_type=model_type, num_classes=5, model_path=model_path)
    # First prepare the data to set class names
    model.prepare(data_path)
    # Then load the model
    model.load_model()
    for image in images:
        prediction = model.predict(image)
        print(f"Prediction for {image}: {prediction}")
    
