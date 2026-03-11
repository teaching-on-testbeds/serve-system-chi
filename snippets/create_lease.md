
::: {.cell .markdown}

## Create a lease for a GPU server

To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with 2x P100 GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

:::

::: {.cell .markdown}

Then,

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_p100` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, on the left side, click on the name of the node you want to reserve.
* Set the "Name" to `serve_system_netID`, replacing `netID` with your actual net ID.
* Set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
* Modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease ten minutes before the end of an hour, e.g. at `YY:50`.
* Click "Next".
* On the "Hosts" tab, confirm that the node you selected is listed in the "Resource properties" section, and click "Next".
* Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.

:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

:::

::: {.cell .markdown}

## At the beginning of your GPU server lease

At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance. To begin this step, open this experiment on Trovi:

* Use this link: [Model optimizations for serving](https://chameleoncloud.org/experiment/share/45097b76-3b24-472d-9b23-d522e795b2e0) on Trovi
* Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it, including the notebook to bring up the bare metal server.

Inside the `serve-system-chi` directory, continue with `2_create_server_nvidia.ipynb`.

:::
