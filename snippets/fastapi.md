

::: {.cell .markdown}

### Benchmarking FastAPI service

Continue here after opening `workspace/4_fastapi.ipynb` in the Jupyter container.


:::


::: {.cell .markdown}


Let's test this service.  First, we'll test the FastAPI endpoint directly. In a browser, run


```
http://A.B.C.D:8000/docs
```

but substitute the floating IP assigned to your instance. This will bring up the [Swagger UI](https://swagger.io/tools/swagger-ui/) associated with the FastAPI endpoint.

Click on "predict" and then "Try it out". Here, we can enter a request to send to the FastAPI endpoint, and see its response.

Our request needs to be in the form of a base64-encoded image. Run

:::


::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
import base64
image_path = "test_image.jpeg"
with open(image_path, 'rb') as f:
    image_bytes = f.read()
encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
print('"' + encoded_str + '"')
```
:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Now that we know everything *works*, let's get some quick performance numbers from this server. We'll send some requests directly to the FastAPI endpoint and measure the time to get a response.

:::


::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
import requests
import time
import numpy as np
```
:::

::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
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
:::


::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / inference_times.sum()  

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} ms")
print(f"Throughput: {throughput:.2f} requests/sec")
```
:::


<!--

Median inference time: 17.2018 ms
95th percentile: 19.4870 ms
99th percentile: 22.2096 ms
Throughput: 57.16 requests/sec

-->


::: {.cell .markdown}

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

to build the FastAPI container image [from the "fastapi_onnx" directory](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/fastapi_onnx/app.py), instead of the "fastapi_pt" directory. 

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

:::



::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
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
:::


::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / inference_times.sum()  

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} ms")
print(f"Throughput: {throughput:.2f} requests/sec")
```
:::




<!--

Median inference time: 9.2471 ms
95th percentile: 11.2387 ms
99th percentile: 16.1481 ms
Throughput: 80.66 requests/sec


-->


::: {.cell .markdown}

Our FastAPI endpoint can maintain low latency, as long as only one user is sending requests to the service.

However, when there are multiple concurrent requests, it will be much slower. For example, suppose we start 16 "senders" at the same time, each continuously sending a new request as soon as it gets a response for the last one:

:::


::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
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
:::



::: {.cell .code}
```python
# runs inside the Jupyter container on node-serve-system
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / total_time

print(f"Median inference time: {1000*median_time:.4f} ms")
print(f"95th percentile: {1000*percentile_95:.4f} ms")
print(f"99th percentile: {1000*percentile_99:.4f} ms")
print(f"Throughput: {throughput:.2f} requests/sec")
```
:::




::: {.cell .markdown}

When a request arrives at the server and finds it busy processing another request, it waits in a queue until it can be served.  This queuing delay can be a significant part of the overall prediction delay, when there is a high degree of concurrency. We will attempt to address this in the next section!

In the meantime, download this entire notebook for later reference.

Then, bring down your current inference service with:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-fastapi.yaml down
```

:::


