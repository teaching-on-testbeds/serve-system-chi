

::: {.cell .markdown}

## Preparing an endpoint in FastAPI

In this section, we will create a FastAPI "wrapper" for our model, so that it can serve inference requests. Once you have finished this section, you should be able to:

* create a FastAPI endpoint for a PyTorch model
* create a FastAPI endpoint for an ONNX model

and run it on CPU or GPU.

:::


::: {.cell .markdown}

### PyTorch version

We have previously seen a [Flask app](https://github.com/teaching-on-testbeds/gourmetgram/blob/master/app.py) that does inference using a pre-trained PyTorch model, and serves a basic browser-based interface for it.

However, to scale up, we will want to separate the model inference service into its own prediction endpoint - that way, we can optimize and scale it separately from the user interface.

[Here is the modified version of the Flask app](https://github.com/teaching-on-testbeds/gourmetgram/blob/fastapi/app.py). Instead of loading a model and making predictions, we send a request to a separate service:

```python
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()
        
        result = response.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")
        
        return predicted_class, probability

    except Exception as e:
        print(f"Error during inference: {e}")  
        return None, None  
```

:::


::: {.cell .markdown}

Meanwhile, [the inference service has moved into a separate app](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/fastapi_pt/app.py):

```python
app = FastAPI(
    title="Food Classification API",
    description="API for classifying food items from images",
    version="1.0.0"
)
# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)  # Ensures probability is between 0 and 1

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Food11 model
MODEL_PATH = "food11.pth"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()

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
    return transform(img).unsqueeze(0)

@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        image = preprocess_image(image).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
            predicted_class = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0, predicted_class].item()  # Get the probability

        return PredictionResponse(prediction=classes[predicted_class], probability=confidence)

    except Exception as e:
        return {"error": str(e)}
```

Let's try it now!

:::

::: {.cell .markdown}

### Bring up containers

To start, run

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml up -d
```

This will use a [Docker Compose file](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/docker/docker-compose-fastapi.yaml) to bring up three containers:

* one container that will host the Flask application, this will serve the web-based user interface of our system
* one container that will host a FastAPI inference endpoint
* and one Jupyter container, which we'll use to run some benchmarking experiments

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container:

```bash
# runs on node-serve-system
docker exec jupyter jupyter server list
```

Look for a line like

```
http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `4_fastapi.ipynb` notebook to continue.


:::
