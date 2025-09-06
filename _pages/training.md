---
layout: distill
title: "Final Run and Training Results"
permalink: /training/
description: "We now write the launch scripts and launch the final run, showcasing how to use multi-controller JAX to conduct large scale, multi-host training runs."
date: 2025-09-06
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: ../moe
previous_section_name: "Part 6: Mixture of Experts"

next_section_url: ../conclusion
next_section_name: "Part 8: Conclusion"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Aditya Makkar
    url: "https://x.com/AdityaMakkar000"
  - name: Divya Makkar
    url: "https://x.com/_DivyaMakkar"
  - name: Chinmay Jindal
    url: "https://x.com/chinmayjindal_"

# Add a table of contents to your post.
#   - make sure that TOC names matcAh the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Final Run Scripts
  - subsections:
      - name: Launcher and Run Scripts
      - name: Cluster Setup and Config
  - name: Training Results
  - subsections:
      - name: Loss and Token Throughput
      - name: Expert Utilization and Auxiliary Loss

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Final Run Scripts

### Launcher and Run Scripts

There are two main scripts that are significant for launching a training run. The first is found in the `launcher.sh` script which contains the IP addresses for all the TPUs as well as a command that launches a training run on each TPU. The command `printf "%s\n" "${IPS[@]}" | xargs -n 1 -P 0 -I {} bash run.sh {}` does the following:

- `printf "%s\n" "${IPS[@]}"` prints each address in the IPS variable on a seperate line
- `| xargs` takes the argument from the ip and runs the command on all distributed devices at once
- `-n 1` runs the command once per input item (each IP gets its own `bash run.sh {}` call)
- `-P 0` runs as many process in parallel where each IP will be processed on a distinct device
- `-I {}` placeholder for the IP argument
- `bash run.sh {}` calls the `run.sh` script passing the IP as an argument

```bash
#!/bin/bash
source .env

IPS=(
    "35.186.25.28"
    "35.186.39.76"
    "107.167.173.215"
    "35.186.132.44"
    "35.186.24.134"
    "35.186.58.69"
    "35.186.134.160"
    "35.186.107.62"
)

printf "%s\n" "${IPS[@]}" | xargs -n 1 -P 0 -I {} bash run.sh {}
```

Essentially the purpose of this script is to execute `run.sh` with each individual IP as an argument to the script, on parallel devices. The purpose of `run.sh` is to:

1. SSH into the IP given as an argument using the command: `ssh $USER@$IP`.
2. Kills any current tmux sessions `tmux kill-session -t $SESSION`, by telling tmux to kill a session with the name matching the `$SESSION` variable.
3. A new tmux session is started `tmux new-session -d -s  $SESSION` with flag `-s $SESSION`, naming the session with the variable name and the flag `-d` creating the session in the background without attaching immediately.
4. Moving to the correct directory and resetting the samples in the folder. This is done with the `tmux send-keys -t $SESSION:0 'cd ~/Jaxformer && rm -rf samples && mkdir samples' C-m` command. `tmux send-keys` tells tmux the keystrokes to execute in the`-t $SESSION:0` in the target session in the first window specified by `:0`. Following that is the actual command to be typed in the session which essentially moves to the Jaxformer directory, removed the folder with samples and then recreates it, essentially resetting the samples. Then `C-m` is executed, which enters the command that was previously typed into the tmux session to run.
5. General setup. The same command as the one above is repeated, except with different internal commands to be executed in the tmux sessions. Specifically, inside the Jaxformer directory, the file is refetched from the git origin and reset to get the latest version. Then, the`setupTPU.sh` script is ran to install the correct dependencies on the TPU, and finally the model is ran as seen in the `$command` variable.

```bash
#!/bin/bash

IP=$1
SESSION="trainingRun"
command="python test.py --checkpoint_steps=75 --n_device_axis 8 2 2 --name moe1B --train_batch_size 32 --use_cache --wandb --eval_steps 10"

echo "Running on $IP"

ssh $USER@$IP "

    tmux kill-session -t $SESSION
    tmux new-session -d -s $SESSION

    tmux send-keys -t $SESSION:0 'cd ~/Jaxformer && rm -rf samples && mkdir samples' C-m
    tmux send-keys -t $SESSION:0 'git fetch origin && git reset --hard origin/main' C-m
    tmux send-keys -t $SESSION:0 'bash setupTpu.sh' C-m
    tmux send-keys -t $SESSION:0 '$command' C-m
"
echo "done commands"
```

### Cluster Setup and Config

For demonstration of the final training, we use the command below which was run across a cluster of 32 TPU-v4 devices across 8 controllers. (8 IPs for ssh).

```bash
python test.py --checkpoint_steps=75 --n_device_axis 8 2 2 --name moe1B --train_batch_size 32 --use_cache --wandb --eval_steps 10"
```

We are using 8 devices for FSDP, 2 for pipeline and 2 for tensor. Here is the final config.

```json
{
  "model_config": {
    "model_dimension": 768,
    "vocab_size": 100277,
    "n_head": 12,
    "blocks": 8,
    "layers_per_block": 6,
    "T": 1024,
    "latent_dim": 128,
    "dhR": 128,
    "dropout_rate": 0.2,
    "model_dtype": "bfloat16",
    "k": 2,
    "n_experts": 16,
    "n_shared": 2,
    "capacity_factor": 1.5
  },
  "data_config": {
    "bucket_name": "350bt_gpt4",
    "process_path": "./bucket_downloads/processShard",
    "train_folder_name": "train",
    "val_folder_name": "val",
    "T": 1024,
    "train_batch_size": 32,
    "val_batch_size": 16,
    "micro_batch_size": 4
  },
  "lr": {
    "max_lr": 0.0006,
    "min_lr": 0,
    "end_lr": 6e-5,
    "warmup_steps": 5000,
    "end_steps": 75000
  },
  "device_config": {
    "n_device_axis": [8, 2, 2]
  },
  "inference_config": {
    "prompt": "hello world",
    "batch_size": 1,
    "top_k": 10000,
    "temperature": 1.0,
    "n_devices": 1,
    "max_tokens": 10,
    "use_cache": true
  },
  "output_dir": "gs://results_jaxformer/",
  "training_steps": 100000,
  "name": "moe1B",
  "grad_step": 1,
  "alpha": 0.0001,
  "checkpoint_steps": 75,
  "eval_steps": 10,
  "seed": 0,
  "wandb": true,
  "grad_clip_norm": 1.0
}
```

In total this config yields 949,248,384 parameters with 343,321,728 active parameters.

We can also see the transformer training across the TPU cluster, showcasing the power of JAX's SPMD paradigm.
{% include figure.liquid path="assets/img/final_run/tpu.png" class="img-fluid" %}


## Training Results

### Loss and Token Throughput

We only train until we hit 3.28 validation loss (inspired by nanoGPT speedrun) due to TRC compute limits. This was achieved after (26,100 steps) and in total $\sim 6.5$ billion tokens; however, with better compute and more time this could continue decreasing.

{% include figure.liquid path="assets/img/final_run/1.png" class="img-fluid" %}

### Expert Utilization and Auxiliary Loss

Notably we avoid expert collapse as seen by the tokens per head and the auxiliary loss curves.

{% include figure.liquid path="assets/img/final_run/2.png" class="img-fluid" %}

{% include figure.liquid path="assets/img/final_run/3.png" class="img-fluid" %}
