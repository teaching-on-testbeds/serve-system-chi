

# System optimizations for serving

We have [previously explored model optimizations for serving](https://teaching-on-testbeds.github.io/serve-model-chi), which focus specifically on reducing the inference time of a model. However, the overall prediction latency of a machine learning system includes other delays besides for that inference time - notably, queuing delay.

In this tutorial, we will explore system-level optimizations to improve those other delay elements. We will:

* learn how to wrap a model in an HTTP endpoint using FastAPI
* and explore system-level optimizations for model serving, including concurrency and batching, in Triton Inference Server


To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.



## Experiment resources 

For this experiment, we will provision one bare-metal node with two NVIDIA P100 GPUs, using a `gpu_p100` node type.



## Create a lease for a GPU server



To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with 2x P100 GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.



Then, 

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_p100` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone. 
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the node name.
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>serve_system_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease five minutes before the end of an hour, e.g. at `YY:55`.
  * Click "Next".
* On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.



Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.



## At the beginning of your GPU server lease


At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! To begin this step, open this experiment on Trovi:

* Use this link: [Model optimizations for serving](https://chameleoncloud.org/experiment/share/45097b76-3b24-472d-9b23-d522e795b2e0) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the bare metal server.






## Launch and set up NVIDIA P100 x2 - with python-chi

At the beginning of the lease time for your bare metal server, we will bring up our GPU instance. We will use the `python-chi` Python API to Chameleon to provision our server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. Also change the site to CHI@TACC or CHI@UC, depending on where your reservation is.


```python
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")
```


Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:


```python
l = lease.get_lease(f"serve_system_netID") 
l.show()
```


The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!



We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-serve-system-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```


Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.



Then, we'll associate a floating IP with the instance, so that we can access it over SSH.


```python
s.associate_floating_ip()
```

```python
s.refresh()
s.check_connectivity()
```


In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).


```python
s.refresh()
s.show(type="widget")
```





### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
s.execute("git clone https://github.com/teaching-on-testbeds/serve-system-chi")
```



### Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and ML training platforms like MLFlow and Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.


```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```


### Set up the NVIDIA container toolkit


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.


```python
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
s.execute("sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json")
s.execute("sudo systemctl restart docker")
```



and, install `nvtop` - 


```python
s.execute("sudo apt -y install nvtop")
```



## Open an SSH session

Finally, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.





## Preparing an endpoint in FastAPI

In this section, we will create a FastAPI "wrapper" for our model, so that it can serve inference requests. Once you have finished this section, you should be able to:

* create a FastAPI endpoint for a PyTorch model
* create a FastAPI endpoint for an ONNX model

and run it on CPU or GPU.




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




Meanwhile, the inference service has moved into a separate app:

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



### Bring up containers

To start, run

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml up -d
```

This will bring up three containers:

* one container that will host the Flask application, this will serve the web-based user interface of our system
* one container that will host a FastAPI inference endpoint
* and one Jupyter container, which we'll use to run some benchmarking experiments

Check the logs of the Jupyter container:

```bash
# runs on node-serve-system
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `fastapi.ipynb` notebook to continue.






Let's test this service.  First, we'll test the FastAPI endpoint directly. In a browser, run


```
http://A.B.C.D:8000/docs
```

but substitute the floating IP assigned to your instance. This will bring up the [Swagger UI](https://swagger.io/tools/swagger-ui/) associated with the FastAPI endpoint.

Click on "predict" and then "Try it out". Here, we can enter a request to send to the FastAPI endpoint, and see its response.

Our request needs to be in the form of a base64-encoded image. Run



```python
import base64
image_path = "test_image.jpeg"
with open(image_path, 'rb') as f:
    image_bytes = f.read()
encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
print('"' + encoded_str + '"')
```


to get the encoded image string. Copy the output of that cell. (After you copy it, you can right-click and clear the cell output, so it won't clutter up the notebook interface.)

Then, in

```
{
  "image": "string"
}
```

replace "string" with the encoded image string you just copied. Press "Execute".

You should see that the server returns a response with code 200 (that's the response code for a successful request) and a response body like:

```
{
  "prediction": "Vegetable/Fruit",
  "probability": 0.9940803647041321
}

```

so we can see that it performed inference successfully on the test input.

Next, let's check the integration of the FastAPI endpoint in our Flask app. In your browser, open

```
http://A.B.C.D
```

but substitute the floating IP assigned to your instance, to access the Flask app. Upload an image and press "Submit" to get its class label.



Now that we know everything *works*, let's get some quick performance numbers from this server. We'll send some requests directly to the FastAPI endpoint and measure the time to get a response.


```python

```



```python
import requests
import time
import numpy as np
```

```python
FASTAPI_URL = "http://fastapi_server:8000/predict"
payload = {"image": encoded_str}
num_requests = 100
inference_times = []

for _ in range(num_requests):
    start_time = time.time()
    response = requests.post(FASTAPI_URL, json=payload)
    end_time = time.time()

    if response.status_code == 200:
        inference_times.append(end_time - start_time)
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
```


```python
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / inference_times.sum()  

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} seconds")
print(f"Throughput: {throughput:.2f} requests/sec")
```


<!--

Median inference time: 17.2018 ms
95th percentile: 19.4870 ms
99th percentile: 22.2096 seconds
Throughput: 57.16 requests/sec

-->



### ONNX version

We know from our previous experiments that the vanilla PyTorch model may not be optimized for inference speed.

Let's try porting our FastAPI endpoint to ONNX.

On the "node-serve-system" host, edit the Docker compose file:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/docker/docker-compose-fastapi.yaml
```

and modify 

```
      context: /home/cc/serve-system-chi/fastapi_pt
```

to 

```
      context: /home/cc/serve-system-chi/fastapi_onnx
```

to build the FastAPI container image from the "fastapi_onnx" directory, instead of the "fastapi_pt" directory. 

Save your changes (Ctrl+O, Enter, Ctrl+X). Rebuild the container image:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml build fastapi_server
```

and recreate the container with the new image:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml up fastapi_server --force-recreate -d
```

Repeat the same steps as before to test the FastAPI endpoint and its integration with Flask. 

Then, re-do our quick benchmark. 

(This host is running an older GPU, so we won't attempt to use the TensorRT execution provider for ONNX, because modern versions no longer support it. So, our results won't be *too* dramatic.)




```python
FASTAPI_URL = "http://fastapi_server:8000/predict"
payload = {"image": encoded_str}
num_requests = 100
inference_times = []

for _ in range(num_requests):
    start_time = time.time()
    response = requests.post(FASTAPI_URL, json=payload)
    end_time = time.time()

    if response.status_code == 200:
        inference_times.append(end_time - start_time)
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
```


```python
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / inference_times.sum()  

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} seconds")
print(f"Throughput: {throughput:.2f} requests/sec")
```




<!--

Median inference time: 9.2471 ms
95th percentile: 11.2387 ms
99th percentile: 16.1481 seconds
Throughput: 80.66 requests/sec


-->



Our FastAPI endpoint can maintain low latency, as long as only one user is sending requests to the service.

However, when there are multiple concurrent requests, it will be much slower. For example, suppose we start 16 "senders" at the same time, each continuously sending a new request as soon as it gets a response for the last one:



```python
import concurrent.futures

def send_request(payload):
    start_time = time.time()
    response = requests.post(FASTAPI_URL, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        return end_time - start_time
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
        return None

def run_concurrent_tests(num_requests, payload, max_workers=10):
    inference_times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                inference_times.append(result)
    
    return inference_times

num_requests = 1000
start_time = time.time()
inference_times = run_concurrent_tests(num_requests, payload, max_workers=16)
total_time = time.time() - start_time
```



```python
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / total_time

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} seconds")
print(f"Throughput: {throughput:.2f} requests/sec")
```





When a request arrives at the server and finds it busy processing another request, it waits in a queue until it can be served.  This queuing delay can be a significant part of the overall prediction delay, when there is a high degree of concurrency. We will attempt to address this in the next section!

In the meantime, bring down your current inference service with:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml down
```






Then, download this entire notebook for later reference.



## Using a Triton Inference Server

[Triton Inference Server](https://developer.nvidia.com/triton-inference-server) is an open-source project by NVIDIA for high-performance ML model deployment. In this section, we will practice deploying models using Triton; after you have finished, you should be able to:

* serve a model using Triton Inference Server with Python backend
* use dynamic batching to improve performance
* scale your model to run on multiple GPUs, and/or with multiple instances on the same GPU
* benchmark the Triton service, and recognize indications of potential problems
* and use optimized backends 



### Anatomy of a Triton model with Python backend

To start, run

```bash
# runs on node-serve-system
cp -r ~/serve-system-chi/models_staging/food_classifier ~/serve-system-chi/models/
```

to copy our first configuration into the directory from which Triton will load models.

Our initial implementation serves our food image classifier using PyTorch. Here's how it works. 

In the Docker compose file, the Triton server is started with the command

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

Let's look at the configuration file first. Here are the contents of `config.pbtxt`:

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

Next, let's look at `model.py`. For a Triton model with Python backend, the `model.py` must define a class named `TritonPythonModel` with at least an `initialize` and `execute` method. Ours has:


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
            transforms.Resize(256),
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

Finally, now that we understand how the server works, let's look at how the Flask app sends requests to it. Inside the Flask app, we now have a function which is called whenever there is a new image uploaded to `predict` or `test`, which sends the image to the Triton server:

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




### Bring up containers


To start, run

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up -d
```

This will bring up three containers:

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

Finally, check the logs of the Jupyter container:

```bash
# runs on node-serve-system
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `triton.ipynb` notebook to continue.

Meanwhile, on the host, run

```bash
# runs on node-serve-system
nvtop
```

to monitor GPU usage - we will refer back to this a few times as we run through the rest of this notebook.





### Serving a PyTorch model


The Triton client comes with a performance analyzer, which we can use to send requests to the server and get some statistics back. Let's try it:


```bash
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 
```



Make a note of the line showing the total average request latency, and the breakdown including:

* `queue`, the queuing delay
* and `compute infer`, the inference delay


<!--

    Avg request latency: 18689 usec (overhead 2 usec + queue 22 usec + compute input 44 usec + compute infer 18570 usec + compute output 49 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 51.549 infer/sec, latency 19311 usec

-->




Let's further exercise this service. In the command above, a single client sends continuous requests to the server - each time a response is returned, a new request is generated. Now, let's configure **8** concurrent clients, each sending continuous requests - as soon as any client gets a response, it sends a new request: 


```bash
# runs inside Jupyter container
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```

<!-- 

    Avg request latency: 151375 usec (overhead 3 usec + queue 132341 usec + compute input 59 usec + compute infer 18922 usec + compute output 49 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 52.3786 infer/sec, latency 151983 usec

-->


Although the inference time (`compute infer`) remains low, the overall system latency is high because of `queue` delay. Only one sample is processed at a time, and other samples have to wait in a queue for their turn. Here, since there are 8 concurrent clients sending continuous requests, the delay is approximately 8x the inference delay. With more concurrent requests, the queuing delay would grow even larger:


```bash
# runs inside Jupyter container
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 16
```

<!-- 

    Avg request latency: 302079 usec (overhead 1 usec + queue 283040 usec + compute input 60 usec + compute infer 18927 usec + compute output 50 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 16, throughput: 52.3609 infer/sec, latency 302804 usec

-->


Although the delay is large (over 100 ms), it's not because of inadequate compute - if you check the `nvtop` display on the host while the test above is running, you will note low GPU utilization! Take a screenshot of the `nvtop` output when this test is running.

We *could* get more throughput without increasing prediction latency, by batching requests:


```bash
# runs inside Jupyter container
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 16 --concurrency-range 1
```

<!--

    Avg request latency: 21189 usec (overhead 3 usec + queue 19 usec + compute input 195 usec + compute infer 20921 usec + compute output 50 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 656.63 infer/sec, latency 24282 usec


-->


But, that's not very helpful in a situation when requests come from individual users, one at a time.





### Dynamic batching

Earlier, we noted that our model can achieve higher throughput with low latency by performing inference on batches of input samples, instead of individual samples. However, our client sends requests with individual samples.

To improve performance, we can ask the Triton server to batch incoming requests whenever possible, and send them through the server together instead of a sequence. In other words, if the server is ready to handle the next request, and it finds four requests waiting in the queue, it should serve those four as a batch instead of just taking the next request in line.




Let's edit the model configuration:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier/config.pbtxt
```

and at the end, add

```
dynamic_batching {
  preferred_batch_size: [4, 6, 8, 10]
  max_queue_delay_microseconds: 100
}

```

Save the file (use Ctrl+O then Enter, then Ctrl+X).

Re-build the container image with this change:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
```

and then bring the server back up:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

and use

```bash
# runs on node-serve-system
docker logs triton_server
```

to make sure the server comes up and is ready. 

Before we benchmark this service again, let's get some pre-benchmark stats about how many requests have been served, broken down by batch size. (If you've just restarted the server, it would be zero!)

```bash
curl http://triton_server:8000/v2/models/food_classifier/versions/1/stats
```



Then, run the benchmark:



```bash
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```

<!--

    Avg request latency: 100423 usec (overhead 6 usec + queue 44892 usec + compute input 197 usec + compute infer 55111 usec + compute output 216 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 78.6276 infer/sec, latency 101232 usec

-->


and get per-batch stats again:


```bash
curl http://triton_server:8000/v2/models/food_classifier/versions/1/stats
```

<!--

{"model_stats":[{"name":"food_classifier","version":"1","last_inference":1741928954242,"inference_count":1436,"execution_count":386,"inference_stats":{"success":{"count":1436,"ns":144129653806},"fail":{"count":0,"ns":0},"queue":{"count":1436,"ns":64542800676},"compute_input":{"count":1436,"ns":283368073},"compute_infer":{"count":1436,"ns":78984688177},"compute_output":{"count":1436,"ns":309635270},"cache_hit":{"count":0,"ns":0},"cache_miss":{"count":0,"ns":0}},"response_stats":{},"batch_stats":[{"batch_size":1,"compute_input":{"count":26,"ns":1754466},"compute_infer":{"count":26,"ns":757012965},"compute_output":{"count":26,"ns":2038319}},{"batch_size":2,"compute_input":{"count":127,"ns":14474588},"compute_infer":{"count":127,"ns":3718519926},"compute_output":{"count":127,"ns":13184875}},{"batch_size":3,"compute_input":{"count":55,"ns":7182962},"compute_infer":{"count":55,"ns":2144383142},"compute_output":{"count":55,"ns":7683505}},{"batch_size":4,"compute_input":{"count":9,"ns":1446080},"compute_infer":{"count":9,"ns":456549788},"compute_output":{"count":9,"ns":1596636}},{"batch_size":5,"compute_input":{"count":73,"ns":14796021},"compute_infer":{"count":73,"ns":4268808423},"compute_output":{"count":73,"ns":16766209}},{"batch_size":6,"compute_input":{"count":82,"ns":19691717},"compute_infer":{"count":82,"ns":5577222019},"compute_output":{"count":82,"ns":22604780}},{"batch_size":7,"compute_input":{"count":14,"ns":4742974},"compute_infer":{"count":14,"ns":1103416079},"compute_output":{"count":14,"ns":4618631}}],"memory_usage":[]}]}

-->


Note that the stats show that requests were served in batch sizes greater than 1, even though each client sent a single request at a time.



### Scaling up

Another easy way to improve performance is to scale up! Let's edit the model configuration:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier/config.pbtxt
```

and change

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
]
```

to

```
  instance_group [
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1 ]
    }
]
```

Save the file (use Ctrl+O then Enter, then Ctrl+X).

Re-build the container image with this change:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
```

and then bring the server back up:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

and use

```bash
# runs on node-serve-system
docker logs triton_server
```

to make sure the server comes up and is ready. 

On the host, run

```bash
# runs on node-serve-system
nvidia-smi
```

and note that there are two instances of `triton_python_backend` processes running on GPU 0, and two on GPU 1.

Then, benchmark *this* service with increased concurrency:


```bash
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```

<!-- 

    Avg request latency: 40707 usec (overhead 3 usec + queue 7036 usec + compute input 75 usec + compute infer 33514 usec + compute output 78 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 192.849 infer/sec, latency 41374 usec

-->



Although there is still some queuing delay (because our degree of concurrency, 8, is still higher than the number of server instances, 4), and the inference time is also increased due to sharing the compute resources, the prediction delay is still on the order of 10s of ms - not over 100ms, like it was previously with concurrency 8!

Also, if you look at the `nvtop` output on the host while running this test, you will observe higher GPU utilization than before (which is good! We want to use the GPU. Underutilization is bad.) However, we are still not fully utilizing the GPU.

Let's try increasing the number of instances again. Edit the model configuration:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier/config.pbtxt
```

and change

```
  instance_group [
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1 ]
    }
]

```

to

```
  instance_group [
    {
      count: 4
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 4
      kind: KIND_GPU
      gpus: [ 1 ]
    }
]
```


Re-build the container image with this change:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
```

and then bring the server back up:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

use

```bash
# runs on node-serve-system
docker logs triton_server
```

to make sure the server comes up and is ready.

Then, re-run our benchmark:


```bash
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```


<!--

    Avg request latency: 66737 usec (overhead 2 usec + queue 466 usec + compute input 61 usec + compute infer 66118 usec + compute output 89 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 118.688 infer/sec, latency 67559 usec

-->


This makes things subtantially worse - our inference time is higher, even though we are still underutilizing the GPU (as seen in `nvtop`).




### Serving an ONNX model

The Python backend we have been using is flexible, but not necessarily the most performant. To get better performance, we will use one of the highly optimized backend in Triton. Since we already have an ONNX model, let's use the ONNX backend.

To serve a model using the ONNX backend, we will create a directory structure like this:

```
food_classifier_onnx/
├── 1
│   └── model.onnx
└── config.pbtxt
```

There is no more `model.py` - Triton serves the model directly, we just have to name it `model.onnx`. In `config.pbtxt`, we will specify the backend as `onnxruntime`:

```
name: "food_classifier_onnx"
backend: "onnxruntime"
max_batch_size: 16
input [
  {
    name: "input"  # has to match ONNX model's input name
    data_type: TYPE_FP32
    dims: [3, 224, 224]  # has to match ONNX input shape
  }
]
output [
  {
    name: "output"  # has to match ONNX model output name
    data_type: TYPE_FP32  # output is a list of probabilities
    dims: [11]  # 
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

Copy this to Triton's models directory:

```bash
# runs on node-serve-system
mkdir ~/serve-system-chi/models/
cp -r ~/serve-system-chi/models_staging/food_classifier_onnx ~/serve-system-chi/models/
```

Re-build the container image with this change:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
```

and then bring the server back up:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

use

```bash
# runs on node-serve-system
docker logs triton_server
```

to make sure the server comes up and is ready. Note that the server will load two models: the original `food_classifier` with Python backend, and the `food_classifier_onnx` model we just added.

Let's benchmark our service. Our ONNX model won't accept image bytes directly - it expects images that already have been pre-processed into arrays. So, our benchmark command will be a little bit different:


```bash
perf_analyzer -u triton_server:8000  -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 
```

<!-- 

    Avg request latency: 4757 usec (overhead 30 usec + queue 26 usec + compute input 117 usec + compute infer 4566 usec + compute output 17 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 138.444 infer/sec, latency 6701 usec

-->


This model has much better inference performance than our PyTorch model with Python backend did, in a similar test. Also, if we monitor with `nvtop`, we should see higher GPU utilization while the test is running (which is a good thing!)

Let's try scaling *this* model up. Edit the model configuration:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier_onnx/config.pbtxt
```

and change

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
]

```

to

```
  instance_group [
    {
      count: 2      
      kind: KIND_GPU
      gpus: [ 0, 1 ]
    }
]
```

and run our benchmark with higher concurrency. (2 instances on each GPU, because we noticed that a single instance used less than half a GPU.) 

Watch the `nvtop` output as you run this test!



```bash
perf_analyzer -u triton_server:8000  -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --concurrency-range 8 
```


<!-- 

    Avg request latency: 3961 usec (overhead 18 usec + queue 697 usec + compute input 97 usec + compute infer 3137 usec + compute output 11 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 1182.39 infer/sec, latency 6089 usec

-->



This time, we should see that our model is fully utilizing the GPU (that's good!) And, our inference performance is much better than the PyTorch model with Python backend could achieve with concurrency 8.

Let's see how we do with even higher concurrency:



```bash
perf_analyzer -u triton_server:8000  -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --concurrency-range 16  
```


<!-- 



    Avg request latency: 9960 usec (overhead 19 usec + queue 6793 usec + compute input 100 usec + compute infer 3036 usec + compute output 11 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 16, throughput: 1257.15 infer/sec, latency 12025 usec

-->


We still have some queue delay, since the rate at which requests arrive is greater than the service rate of the models. But, we can feel good that we are no longer underutilizing the GPUs!




There's one more issue we should address: our ONNX model doesn't directly work with our Flask server now, because the inputs and outputs are different. The ONNX model expects a pre-processed array, and returns a list of class probabilities. 

Since the pre-processing and post-processing doesn't need GPU anyway, we'll move it to the Flask app.

Edit the Docker compose file:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/docker/docker-compose-triton.yaml
```


and change

```
  flask:
    build:
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#triton
```

to 

```
  flask:
    build:
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#triton_onnx
```

to use a version of our Flask app where the pre- and post-processing is built in. Also change

```
      - FOOD11_MODEL_NAME=food_classifier
```

to 

```
      - FOOD11_MODEL_NAME=food_classifier_onnx
```

so that our Flask app will send requests to the correct model.

Then run

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build flask
```

to re-build the container image, and

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up flask --force-recreate -d
```

to restart the Flask container with the new image.

Let's test this service.  In a browser, run

```
http://A.B.C.D
```

but substitute the floating IP assigned to your instance, to access the Flask app. Upload an image and press "Submit" to get its class label.




Then, download this entire notebook for later reference.



<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>