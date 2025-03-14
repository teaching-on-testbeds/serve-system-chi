
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

## Create a lease for a GPU server

:::

::: {.cell .markdown}

To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with 2x P100 GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

:::

::: {.cell .markdown}

## At the beginning of your GPU server lease


At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! To begin this step, open this experiment on Trovi:

* Use this link: [Model optimizations for serving](https://chameleoncloud.org/experiment/share/45097b76-3b24-472d-9b23-d522e795b2e0) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the bare metal server.


:::
