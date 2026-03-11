
::: {.cell .markdown}

# System optimizations for serving

We have [previously explored model optimizations for serving](https://teaching-on-testbeds.github.io/serve-model-chi), which focus specifically on reducing the inference time of a model. However, the overall prediction latency of a machine learning system includes other delays besides for that inference time - notably, queuing delay.

In this tutorial, we will explore system-level optimizations to improve those other delay elements. We will:

* learn how to wrap a model in an HTTP endpoint using FastAPI
* and explore system-level optimizations for model serving, including concurrency and batching, in Triton Inference Server


To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with two NVIDIA P100 GPUs, using a `gpu_p100` node type.

:::

::: {.cell .markdown}

Continue with `1_create_lease.ipynb`.

:::
