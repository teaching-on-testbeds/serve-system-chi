

::: {.cell .markdown}

## Using a Triton Inference Server

[Triton Inference Server](https://developer.nvidia.com/triton-inference-server) is an open-source project by NVIDIA for high-performance ML model deployment. In this section, we will practice deploying models using Triton; after you have finished, you should be able to:

* serve a model using Triton Inference Server with Python backend
* use dynamic batching to improve performance
* scale your model to run on multiple GPUs, and/or with multiple instances on the same GPU
* benchmark the Triton service, and recognize indications of potential problems
* and use optimized backends 

:::

::: {.cell .markdown}

### Anatomy of a Triton model with Python backend

To start, run

```bash
# runs on node-serve-system
mkdir ~/serve-system-chi/models/
cp -r ~/serve-system-chi/models_staging/food_classifier ~/serve-system-chi/models/
```

to copy [our first configuration](https://github.com/teaching-on-testbeds/serve-system-chi/tree/main/models_staging/food_classifier) into the directory from which Triton will load models.

Our initial implementation serves our food image classifier using PyTorch. Here's how it works. 

In the [Dockerfile](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/docker/Dockerfile.triton), the Triton server is started with the command

```bash
tritonserver --model-repository=/models
```

where the `/models` directry is organized as follows:

```
models/
└── food_classifier
    ├── 1
    │   ├── food11.pth
    │   └── model.py
    └── config.pbtxt
```

It includes:

* a top-level directory whose name is the "model name"
* a configuration file `config.pbtxt` inside that directory. We'll look at that shortly.
* and a subdirectory for each model version. We have model version 1, so we have a subdirectory 1. Inside this directory is a `model.py`, which describes how the model will run.

Let's [look at the configuration file first](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/models_staging/food_classifier/config.pbtxt). Here are the contents of `config.pbtxt`:

```
name: "food_classifier"
backend: "python"
max_batch_size: 16
input [
  {
    name: "INPUT_IMAGE"
    data_type: TYPE_STRING
    dims: [1]
  }
]
output [
  {
    name: "FOOD_LABEL"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "PROBABILITY"
    data_type: TYPE_FP32
    dims: [1]
  }
]
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
]
```

We have defined:

* a `name`, which must match the directory name
* a `backend` - we are using the basic [Python backend](https://github.com/triton-inference-server/python_backend). This is a highly flexible backend which allows us to define how our model will run by providing Python code in a `model.py` file. 
* a `max_batch_size` - we have set it to 16, but generally you would set this according to the GPU memory available
* the `name`, `data_type`, and `dims` (dimensions) of each `input` to the model
* the `name`, `data_type`, and `dims` (dimensions) of each `output` from the model
* an `instance_group` with the `count` (number of copies of the model that we want to serve) and details of the device we want to serve it on (we will serve it on GPU 0). Note that to run the model on CPU instead, we could have used

```
  instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
  ]
```

Next, let's [look at `model.py`](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/models_staging/food_classifier/1/model.py). For a Triton model with Python backend, the `model.py` must define a class named `TritonPythonModel` with at least an `initialize` and `execute` method. Ours has:


* An `initialize` method to load the model, move it to the device specified in the `args` passed from the Triton server, and put it in inference mode. This will run as soon as Triton starts and loads models from the directory passed to it:

```python
def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "food11.pth")
        
        # From args, get info about what device the model is supposed to be on
        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.classes = np.array([
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ])

```

* A `preprocess` method, which will run on each input image that is passed:

```python
def preprocess(self, image_data):
    if isinstance(image_data, str):
        image_data = base64.b64decode(image_data)

    if isinstance(image_data, bytes):
        image_data = image_data.decode("utf-8")
        image_data = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    img_tensor = self.transform(image).unsqueeze(0)
    return img_tensor
```

* and an `execute`, which will apply to batches of requests sent to this model:

```python
def execute(self, requests):
    # Gather inputs from all requests
    batched_inputs = []
    for request in requests:
        in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
        input_data_array = in_tensor.as_numpy()  # each assumed to be shape [1]
        # Preprocess each input (resulting in a tensor of shape [1, C, H, W])
        batched_inputs.append(self.preprocess(input_data_array[0, 0]))
    
    # Combine inputs along the batch dimension
    batched_tensor = torch.cat(batched_inputs, dim=0).to(self.device)
    print("BatchSize: ", len(batched_inputs))
    # Run inference once on the full batch
    with torch.no_grad():
        outputs = self.model(batched_tensor)
    
    # Process the outputs and split them for each request
    responses = []
    for i, request in enumerate(requests):
        output = outputs[i:i+1]  # select the i-th output
        prob, predicted_class = torch.max(output, 1)
        predicted_label = self.classes[predicted_class.item()]
        probability = torch.sigmoid(prob).item()
        
        # Create numpy arrays with shape [1, 1] for consistency.
        out_label_np = np.array([[predicted_label]], dtype=object)
        out_prob_np = np.array([[probability]], dtype=np.float32)
        
        out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
        out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)
        
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_label, out_tensor_prob])
        responses.append(inference_response)
    
    return responses
```

Finally, now that we understand how the server works, let's [look at how the Flask app sends requests to it](https://github.com/teaching-on-testbeds/gourmetgram/blob/triton/app.py). Inside the Flask app, we now have a function which is called whenever there is a new image uploaded to `predict` or `test`, which sends the image to the Triton server:

```python
def request_triton(image_path):
    try:
        # Connect to Triton server
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        # Prepare inputs and outputs
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))

        encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput("FOOD_LABEL", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("PROBABILITY", binary_data=False))

        # Run inference
        results = triton_client.infer(model_name=FOOD11_MODEL_NAME, inputs=inputs, outputs=outputs)

        predicted_class = results.as_numpy("FOOD_LABEL")[0,0]
        probability = results.as_numpy("PROBABILITY")[0,0]

        return predicted_class, probability

    except Exception as e:
        print(f"Error during inference: {e}")  
        return None, None  
```


:::

::: {.cell .markdown}

### Bring up containers


To start, run

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up -d
```

This uses a [Docker Compose configuration](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/docker/docker-compose-triton.yaml) to bring up three containers:

* one container with NVIDIA Triton Server, with the host's GPUs passed to the container, and with the `models` directory (containing the model and its configuration) passed as a bind mount
* one container that hosts the Flask app, which will serve the user interface and send inference requests to the Triton server
* one Jupyter container with the Triton client installed, for us to conduct a performance evaluation of the Triton server


Watch the logs from the Triton server as it starts up:

```bash
# runs on node-serve-system
docker logs triton_server -f
```

Once the Triton server starts up, you should see something like

```
+--------------------------+---------+--------+
| Model                    | Version | Status |
+--------------------------+---------+--------+
| food_classifier | 1       | READY  |
+--------------------------+---------+--------+
```

and then some additional output. Near the end, you will see

```
"Started GRPCInferenceService at 0.0.0.0:8001"
"Started HTTPService at 0.0.0.0:8000"
"Started Metrics Service at 0.0.0.0:8002"
```

(and then some messages about not getting GPU power consumption, which is fine and not a concern.)

You can use Ctrl+C to stop watching the logs once you see this output.

Let's test this service.  In a browser, run

```
http://A.B.C.D
```

but substitute the floating IP assigned to your instance, to access the Flask app. Upload an image and press "Submit" to get its class label.

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

Then, in the file browser on the left side, open the "work" directory and then click on the `6_triton.ipynb` notebook to continue.

Meanwhile, on the host, run

```bash
# runs on node-serve-system
nvtop
```

to monitor GPU usage - we will refer back to this a few times as we run through the rest of this notebook.

:::
