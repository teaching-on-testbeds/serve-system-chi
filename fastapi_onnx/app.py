from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
import onnxruntime as ort  # ONNX Runtime
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(
    title="Food Classification API (ONNX)",
    description="API for classifying food items from images using ONNX Runtime",
    version="1.0.0"
)

# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)  # Ensures probability is between 0 and 1

# Load ONNX model
MODEL_PATH = "food11.onnx"

# Set device (Use GPU if available, otherwise CPU)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# Define class labels
classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"])

# Define the image preprocessing function
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).numpy()  # Convert to NumPy for ONNX
    return img

@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        image = preprocess_image(image)

        # Run inference with ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})

        probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]))  # Softmax manually
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[0, predicted_class_idx]

        return PredictionResponse(prediction=classes[predicted_class_idx], probability=float(confidence))

    except Exception as e:
        return {"error": str(e)}
