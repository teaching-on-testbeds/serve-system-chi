

::: {.cell .markdown}

### Benchmarking Triton service


Continue here after opening `workspace/6_triton.ipynb` in the Jupyter container.

:::

::: {.cell .markdown}

### Serving a PyTorch model


The Triton client comes with a performance analyzer, which we can use to send requests to the server and get some statistics back. Let's try it:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 
```
:::


::: {.cell .markdown}

Make a note of the line showing the total average request latency, and the breakdown including:

* `queue`, the queuing delay
* and `compute infer`, the inference delay

:::

<!--

    Avg request latency: 18689 usec (overhead 2 usec + queue 22 usec + compute input 44 usec + compute infer 18570 usec + compute output 49 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 51.549 infer/sec, latency 19311 usec

-->



::: {.cell .markdown}

Let's further exercise this service. In the command above, a single client sends continuous requests to the server - each time a response is returned, a new request is generated. Now, let's configure **8** concurrent clients, each sending continuous requests - as soon as any client gets a response, it sends a new request: 

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```
:::

<!-- 

    Avg request latency: 151375 usec (overhead 3 usec + queue 132341 usec + compute input 59 usec + compute infer 18922 usec + compute output 49 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 52.3786 infer/sec, latency 151983 usec

-->

::: {.cell .markdown}

While the inference time (`compute infer`) is similar to the previous example, the overall system latency is high because of `queue` delay. Only one sample is processed at a time, and other samples have to wait in a queue for their turn. Here, since there are 8 concurrent clients sending continuous requests, the delay is approximately 8x the inference delay. 


With more concurrent requests, the queuing delay would grow even larger:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 16
```
:::

<!-- 

    Avg request latency: 302079 usec (overhead 1 usec + queue 283040 usec + compute input 60 usec + compute infer 18927 usec + compute output 50 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 16, throughput: 52.3609 infer/sec, latency 302804 usec

-->

::: {.cell .markdown}

Although the delay is large (over 100 ms), it's not because of inadequate compute - if you check the `nvtop` display on the host while the test above is running, you will note low GPU utilization! Take a screenshot of the `nvtop` output when this test is running.

We *could* get more throughput without increasing prediction latency, by batching requests. Here, we have a single client sending requests in batches of 16 at a time:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 16 --concurrency-range 1
```
:::

<!--

    Avg request latency: 21189 usec (overhead 3 usec + queue 19 usec + compute input 195 usec + compute infer 20921 usec + compute output 50 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 656.63 infer/sec, latency 24282 usec


-->

::: {.cell .markdown}

We can see that a batch of 16 requests doesn't have much higher inference time than a single request. The throughput is substantially higher when we can serve in batches.

But, that's not very helpful in a situation when requests come from individual users, one at a time.
:::



::: {.cell .markdown}

### Scaling up PyTorch model

One potential way to improve performance is to scale up! Let's edit the model configuration:

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

to run two instances on GPU 0 and two instances on GPU 1:

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

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```
:::

<!-- 

    Avg request latency: 40707 usec (overhead 3 usec + queue 7036 usec + compute input 75 usec + compute infer 33514 usec + compute output 78 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 192.849 infer/sec, latency 41374 usec

-->


::: {.cell .markdown}

There is still *some* queuing delay (because our degree of concurrency, 8, is still higher than the number of server instances, 4), and furthermore, the inference time is also increased due to sharing the compute resources. However, the prediction delay is on the order of 10s of ms - not over 100ms, like it was previously with concurrency 8!

Also, if you look at the `nvtop` output on the host while running this test, you will observe higher GPU utilization than before (which is good! We want to use the GPU. Underutilization is bad.) (Take a screenshot!) However, we are still not fully utilizing the GPU.

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

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier  --input-data input.json -b 1 --concurrency-range 8
```
:::


<!--

    Avg request latency: 66737 usec (overhead 2 usec + queue 466 usec + compute input 61 usec + compute infer 66118 usec + compute output 89 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 118.688 infer/sec, latency 67559 usec

-->

::: {.cell .markdown}

This makes things worse - our inference time is higher, even though we are still underutilizing the GPU (as seen in `nvtop`) (take a screenshot!). 

Our system is not limited by GPU - we are underutilizing the GPU. However, we are being killed by the overhead of the Python backend and our `model.py` implementation.


:::

::: {.cell .markdown}

### Serving an ONNX model

The Python backend we have been using is flexible, but not necessarily the most performant. To get better performance, we will use one of the highly optimized backend in Triton. Since we already have an ONNX model, let's use the ONNX backend.

To serve a model using the ONNX backend, we will create a [directory structure like this](https://github.com/teaching-on-testbeds/serve-system-chi/tree/main/models_staging/food_classifier_onnx):

```
food_classifier_onnx/
├── 1
│   └── model.onnx
└── config.pbtxt
```

There is no more `model.py` - Triton serves the model directly, we just have to name it `model.onnx`. In [`config.pbtxt`](https://github.com/teaching-on-testbeds/serve-system-chi/blob/main/models_staging/food_classifier_onnx/config.pbtxt), we will specify the backend as `onnxruntime`:

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

and use

```bash
# runs on node-serve-system
docker logs triton_server
```

to make sure the server comes up and is ready. Note that the server will load two models: the original `food_classifier` with Python backend, and the `food_classifier_onnx` model we just added.

Let's benchmark our service. Our ONNX model won't accept image bytes directly - it expects images that already have been pre-processed into arrays. So, our benchmark command will be a little bit different:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --concurrency-range 1 
```
:::

<!-- 

    Avg request latency: 4757 usec (overhead 30 usec + queue 26 usec + compute input 117 usec + compute infer 4566 usec + compute output 17 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 138.444 infer/sec, latency 6701 usec

-->

::: {.cell .markdown}

This model has much better inference performance than our PyTorch model with Python backend did, in a similar test. Also, if we monitor with `nvtop`, we should see higher GPU utilization while the test is running (which is a good thing!) (Take a screenshot!)

:::

::: {.cell .markdown}

### Scaling up ONNX model

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

Then, run our benchmark with higher concurrency. (2 instances on each GPU, because we noticed that a single instance used less than half a GPU.) 

(Note that in this example and the following one, we limit the number of requests sent by `perf_analyzer`; this is necessary because of measurement instability under high concurrency.)

Watch the `nvtop` output as you run this test! (Take a screenshot!)


:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000  -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --concurrency-range 8 --warmup-request-count 500 --request-count 20000
```
:::


<!-- 

    Avg request latency: 3943 usec (overhead 19 usec + queue 674 usec + compute input 93 usec + compute infer 3144 usec + compute output 11 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 8, throughput: 1110.43 infer/sec, latency 6252 usec


-->


::: {.cell .markdown}

This time, we should see that our model is fully utilizing the GPU (that's good!) And, our system performance is much better than the PyTorch model with Python backend could achieve with concurrency 8. We still have very little queuing delay.

Let's see how we do with even higher concurrency.

:::


::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --concurrency-range 16 --warmup-request-count 500 --request-count 20000
```
:::


<!-- 



Avg request latency: 10162 usec (overhead 18 usec + queue 6872 usec + compute input 123 usec + compute infer 3136 usec + compute output 11 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 16, throughput: 1175.41 infer/sec, latency 12729 usec

-->

::: {.cell .markdown}

We still have some queuing delay - the average request waits longer in the queue than its actual service time! - since the rate at which requests arrive is greater than the service rate of the models.

But, we can feel good that we are no longer underutilizing the GPUs (as evidenced by `nvtop` output)!

:::

::: {.cell .markdown}


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

to use [a version of our Flask app where the pre- and post-processing is built in](https://github.com/teaching-on-testbeds/gourmetgram/blob/triton_onnx/app.py). Also change

```
      - FOOD11_MODEL_NAME=food_classifier
```

to 

```
      - FOOD11_MODEL_NAME=food_classifier_onnx
```

so that our Flask app will send requests to the new ONNX model service.

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


:::


::: {.cell .markdown}

### Dynamic batching with ONNX model

Until now, we have been working to reduce delay when there is a high, but steady, flow of requests arriving at the service.

In most realistic cases, however, the rate at which requests arrive is variable. Some time may pass with only a couple of requests, and then suddenly a burst of requests arrive. This is more challenging, because the same average request rate that is easily served with a constant interarrival pattern can have queuing delay when the arrivals are bursty.

Let us explore this further in this section.

:::

::: {.cell .markdown}

First, open the config

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier_onnx/config.pbtxt
```

and let's change back

```
  instance_group [
    {
      count: 2      
      kind: KIND_GPU
      gpus: [ 0, 1 ]
    }
]
```

to

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
]

```

so we will work with just one model instance again.

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

:::


::: {.cell .markdown}

Now we will benchmark with `perf_analyzer` again. But,

* instead of scaling up load with a higher `--concurrency-range`, we will scale with `--request-rate-range` (which defines the average number of requests per second), 
* and we can vary the `--request-distribution` between `constant` interarrival time and `poisson`. 

(Note: when we set a request rate, the throughput will never be higher than that rate, since throughput measures requests served per second. We will ignore these throughput measurements, since they reflect the request pattern and not the server capacity.)

Let's first try sending 120 requests per second with a constant interarrival pattern. We know from our earlier tests that with one model instance, the server is still capable of processing requests at this rate:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 120 --request-distribution constant
```
:::

<!--

    Avg request latency: 5476 usec (overhead 35 usec + queue 37 usec + compute input 139 usec + compute infer 5244 usec + compute output 20 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 120, throughput: 120.027 infer/sec, latency 6851 usec


-->

::: {.cell .markdown}

Then, repeat with a Poisson arrival process:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 120 --request-distribution poisson
```
:::

<!--

    Avg request latency: 7314 usec (overhead 30 usec + queue 2722 usec + compute input 116 usec + compute infer 4428 usec + compute output 18 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 120.00, throughput: 116.51 infer/sec, latency 9731 usec

-->

::: {.cell .markdown}

With Poisson arrivals at the same average rate, requests sometimes arrive in bursts and sometimes with gaps. The bursts cause queue buildup, leading to much queue delay even though the average rate is the same.

:::

::: {.cell .markdown}

This problem is not as easily addressed by provisioning more instances. Scaling out instances for bursty traffic is expensive and still leaves servers underutilized between spikes.  Instead, we will try dynamic batching.

Earlier, we noted that our model can achieve higher throughput with low latency by performing inference on batches of input samples, instead of individual samples. But, our client sends requests with individual samples. 

When requests arrive in a burst and are queued, however, we can batch them and then send them to the server as a batch, instead of in sequence. In other words, if the server is ready to handle the next request, and it finds four requests waiting in the queue, it should serve those four as a batch instead of just taking the next request in line. This approach absorbs short-term request bursts without constant overprovisioning.

:::


::: {.cell .markdown}

Let's edit the model configuration:

```bash
# runs on node-serve-system
nano ~/serve-system-chi/models/food_classifier_onnx/config.pbtxt
```

and at the end, add

```
dynamic_batching {
  preferred_batch_size: [4, 6, 8]
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

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
curl -s http://triton_server:8000/v2/models/food_classifier_onnx/versions/1/stats | python -m json.tool
```
:::


::: {.cell .markdown}

Then, run the benchmark again with Poisson arrivals:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 120 --request-distribution poisson
```
:::



<!--

Avg request latency: 7225 usec (overhead 32 usec + queue 2074 usec + compute input 160 usec + compute infer 4939 usec + compute output 18 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 120.00, throughput: 116.44 infer/sec, latency 9578 usec

-->

::: {.cell .markdown}

and get per-batch stats again:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
curl -s http://triton_server:8000/v2/models/food_classifier_onnx/versions/1/stats | python -m json.tool
```
:::

<!--

batch_stats": [
                {
                    "batch_size": 1,
                    "compute_input": {
                        "count": 2110,
                        "ns": 250195975
                    },
                    "compute_infer": {
                        "count": 2110,
                        "ns": 16088167166
                    },
                    "compute_output": {
                        "count": 2110,
                        "ns": 38857833
                    }
                },
                {
                    "batch_size": 2,
                    "compute_input": {
                        "count": 790,
                        "ns": 208620821
                    },
                    "compute_infer": {
                        "count": 790,
                        "ns": 4119542417
                    },
                    "compute_output": {
                        "count": 790,
                        "ns": 11307106
                    }
                },
                {
                    "batch_size": 3,
                    "compute_input": {
                        "count": 30,
                        "ns": 16268328
                    },
                    "compute_infer": {
                        "count": 30,
                        "ns": 1523793977
                    },
                    "compute_output": {
                        "count": 30,
                        "ns": 631592
                    }
                }
            ],
-->

::: {.cell .markdown}

Observe that the stats show that some requests were served in batch sizes greater than 1, even though each client sent a single request at a time.

:::

::: {.cell .markdown}

When the average queuing delay is still low, we may not see much improvement in overall latency due to dynamic batching. Under these circumstances, even with dynamic batching on, a request that arrives while the server is busy will still have to wait (on average) for half of an inference time. But, watch what happens when we scale up the request rate:

:::

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 180 --request-distribution poisson
```
:::

<!--

Avg request latency: 5807 usec (overhead 26 usec + queue 1820 usec + compute input 142 usec + compute infer 3803 usec + compute output 14 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 180.00, throughput: 174.95 infer/sec, latency 8237 usec

-->

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 240 --request-distribution poisson
```
:::


<!--

Avg request latency: 5040 usec (overhead 22 usec + queue 1650 usec + compute input 137 usec + compute infer 3218 usec + compute output 12 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 240.00, throughput: 238.57 infer/sec, latency 7509 usec

-->

::: {.cell .code}
```bash
# runs inside the Jupyter container on node-serve-system
perf_analyzer -u triton_server:8000 -m food_classifier_onnx -b 1 --shape IMAGE:3,224,224 --request-rate-range 300 --request-distribution poisson
```
:::

<!--

Avg request latency: 5098 usec (overhead 24 usec + queue 1764 usec + compute input 155 usec + compute infer 3142 usec + compute output 12 usec)
Inferences/Second vs. Client Average Batch Latency
Request Rate: 300.00, throughput: 294.63 infer/sec, latency 7302 usec

-->

::: {.cell .markdown}

Even as we increase the request rate, the average request will still only wait half of a service time, because once the request that is currently in service finishes, every request waiting in the queue is processed as a batch.

(In fact, we may even see *less* overall latency for higher request rates, because the GPU remains "warm".)


:::



::: {.cell .markdown}


When you have finished, download this entire notebook for later reference.

Then, bring down your current inference service with:

```bash
# runs on node-serve-system
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml down
```


:::
