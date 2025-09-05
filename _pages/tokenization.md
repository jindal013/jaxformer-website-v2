---
layout: distill
title: "Tokenization at Scale"
permalink: /tokenization/
description: "This section describes how to efficiently tokenize large amounts of text via distributed computing on CloudTPUs and Python multiprocessing. We also expose an interface for shard checkpointing to handle unexpected interruptions in data uploading to GCP buckets. The script is adapted from Andrej Karpathy's NanoGPT project with optimizations to process data at a larger scale."
date: 2025-09-05
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Part 0: Introduction"

next_section_url: ../base_model
next_section_name: "Part 2: Base Model"

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
  - name: Introduction
  - subsections:
      - name: "Libraries, Tokenizer & Dataset"
      - name: "Splits & Streaming"
  - name: Multiprocessing on a Single VM
  - subsections:
      - name: "Core concepts"
      - name: "Pool() Multiprocessing"
  - name: Integration with GCP
  - subsections:
      - name: "Single-use Folder Script"
      - name: "Uploading to GCP Buckets"
  - name: Shard Checkpointing
  - name: Distributed Multiprocessing using Ray
  - subsections:
      - name: "Cluster Setup"
      - name: "Adding Ray"
      - name: "Final Script"

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

## Introduction

The tokenization script was built off of Andrej Karpathy's [Build-NanoGPT](https://github.com/karpathy/build-nanogpt) architecture with quite a few major changes. Let's first briefly discuss the basics of the original script before moving on to our additions which significantly speed up the process.

{% include figure.liquid path="assets/img/tokenization/1.png" class="img-fluid" caption="Tokenization of a sentence into 'tokens'. Example with GPT-2 tokenizer" %}

Tokenization is the process of breaking text (in our case, UTF-8 encoding) into smaller chunks that can be used to form a finitely sized vocabulary for an LLM. The exact process for deciding between the tradeoffs of vocab size and average character length of a token (eg. splitting text into individual characters yields a smaller vocab, but loses more information vs individual words) is not done manually. The tokenizer uses the [Byte-Pair Encoding](https://huggingface.co/learn/llm-course/en/chapter6/5) (BPE) algorithm, which is tested and optimized differently for various models.

### Libraries, Tokenizer & Dataset

The tokenizer is pre-trained and loaded through OpenAI's [tiktoken](https://github.com/openai/tiktoken) library. Tiktoken is a fast BPE tokenizer that is used with OpenAI's models. We use the GPT-4 tokenizer (`cl100k_base`) with a vocab size of `100,277` and thus the `uint32` data type is used. The tokenize function grabs the `"text"` value of each dataset row and converts it into a `numpy` array. The `doc_id` is returned for checkpointing purposes, which will be explained in more detail below.

```python
enc = tiktoken.encoding_for_model("gpt-4") # 'cl100k_base'
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
  doc_id = doc['id']
  tokens = [eot] # the special <|endoftext|> token delimits all documents
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
  tokens_np_uint32 = tokens_np.astype(np.uint32)
  return tokens_np_uint32, doc_id
```

Python's native [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module was used for spawning multiple worker processes that each call the tokenize function. This was implemented for a single machine, meaning that the script could only utilize the max CPUs provided by a single instance. We also utilize this for single VM multiprocessing. However, for a distributed implementation, [Ray](https://docs.ray.io/en/latest/index.html) was utilized to create a cluster across multiple machines. The code snippet below starts the Pooling processing with the `mp.Pool()` context manager. The parameter, `nprocs = int(os.cpu_count()) // 2`, denotes the number of worker processes to independently start. Floor division by 2 prevents over-saturating CPUs with too many workers, ensuring smoother performance and less contention.

```python
with mp.Pool(nprocs) as pool:
  ...
  # preallocate buffer to hold current shard
  all_tokens_np = np.empty((shard_size,), dtype=np.uint32)

  for tokens, doc_id in pool.imap(tokenize, fw, chunksize=16):

    # check if current shard can accomodate new tokens
    # if yes --> simply append
    # if not --> write current shard to file, checkpoint, start new

    # at the end --> fill last shard and write remaining to new file
    ...
```

HuggingFace's [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset was used. The original script used the Sample-10BT bucket, a subset randomly sampled from the whole dataset of around 10B gpt2 tokens. Our modified script uses the Sample-350BT bucket as we aimed to launch much larger training runs. The `load_dataset()` data loader from HuggingFace [datasets](https://huggingface.co/docs/datasets/en/index) API was utilized.

```python
remote_name = "sample-350BT"
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name)
```

### Splits & Streaming

A more robust method for changing test and train splits was added. This was done simply by keeping a global variable, `TEST_SPLIT` which would indicate the shard at which you wish to stop each split, assuming the order is 1) test and 2) train for the remaining shards. Then, during tokenization, the `shard_index` variable was used to track which shard the script was on. Simple conditional logic was added to then redirect the shard to the appropriate GCP bucket, update it's naming convention and also the uploaded `shard_index_number` (different from `shard_index`) so that it resets every split.

```python
# 90:10 train, test split
TEST_SPLIT = 350 # 0 (inclusive) to 350 (exclusive) shards are test
# rest are train

...
for tokens, doc_id in pool.imap(tokenize, fw, chunksize=16):
    ...

  if shard_index >= 0 and shard_index < TEST_SPLIT:
        split = 'test/'
        shard_index_number = shard_index
    else:
      split = 'train/'
      shard_index_number = shard_index - TEST_SPLIT
    split_name = split[:-1]
...
```

Another design decision was to stream the Hugging Face (HF) dataset. Streaming a HF dataset means progressively loading and processing data as you iterate, without downloading the entire dataset to disk. This is ideal for our use case as we can start tokenizing shards right away.

```python
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
```

## Multiprocessing on a Single VM

A single VM can support multiprocessing as it has multiple CPU cores. We can utilize each of them by spawning identical tokenization processes on each so that it can be done in parallel.

### Core concepts

Tokenization is a CPU-bound task, which makes Python’s normal threading ineffective because of the [Global Interpreter Lock (GIL)](https://en.wikipedia.org/wiki/Global_interpreter_lock).<d-footnote>Note: Starting in Python 3.13, this limitation has been lifted via an experimental free-threaded build that disables the GIL.</d-footnote> GIL makes it so that only one thread can execute Python bytecode at a time. For I/O tasks or API requests, threading is fine, but for heavy computation it gives almost no speedup as it provides concurrency and not true parallelism.

{% include figure.liquid path="assets/img/tokenization/2.png" class="img-fluid" caption="Multiprocessing uses separate processes whereas threading handles concurrency (<a href='https://datanoon.com/blog/multiprocessing_in_python/'>source</a>)" %}

This is why we switch to multiprocessing: each worker runs in its own process, bypassing the GIL and truly using multiple CPU cores in parallel. Each worker independently runs `tokenize()`, while the main process orchestrates shard writing and uploads.

### Pool() Multiprocessing

A `Pool` is just a convenience wrapper in Python’s `multiprocessing` module that manages a group of worker processes for you. Instead of manually creating and tracking processes, you create a pool, give it a function (like `tokenize()`), and it will distribute work across the workers.

```python
with mp.Pool(nprocs) as pool:
    for tokens, doc_id in pool.imap(tokenize, fw, chunksize=16):
        ...
```

In the code, the `Pool` spins up `nprocs` worker processes and each worker runs independently on a CPU core. The `pool.imap()` function is similar to `map()` except that it returns back an iterator so the main process can keep writing shards while workers continue tokenizing. In essence, you can start receiving results from workers as soon as they're ready with `imap`, rather than having to wait for all of them to be finished. The `chunksize` parameter will cause the iterable to be split into pieces of approximately that size, and each piece is submitted as a separate task. Other aspects of the script include writing the shards to the file, and then a for loop to append shards until the desired size (`100M`) is reached, after which it is stored and a new shard/file is started. Progress bar tracking has been taken out of the code snippet below to improve readability.

```python
# function to save sharded file to local disk
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

...

with mp.Pool(nprocs) as pool:
    shard_index = 0 # current shard index

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0

    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # if there is enough space in the current shard
        if token_count + len(tokens) < shard_size:
            # append tokens to current shard
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"{split}_{shard_index:06d}"
            )
            # fill the remaining document, then start new shard
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1

            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
```

## Integration with GCP

For this project, Google Cloud Storage (GCS) was used due to it's strong integration in the JAX ecosystem. In order to create a bucket with support for folders, the `Hierarchical namespace` was enabled in the GC Console after starting a new project.

### Single-use Folder Script

After creation, [the TPU can be authenticated](https://github.com/jindal013/gcp_tokenizer/blob/main/README.md) so that it can read/write to the bucket. Now, we run the following script to create the checkpoints, train, and test folders. We use with the the Python Client API for GCS.

```python
from google.cloud import storage_control_v2

def create_folder(bucket_name: str, folder_name: str) -> None:
    storage_control_client = storage_control_v2.StorageControlClient()
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    request = storage_control_v2.CreateFolderRequest(
        parent=bucket_path,
        folder_id=folder_name,
    )
    response = storage_control_client.create_folder(request=request)

    print(f"Created folder: {response.name}")

if __name__ == '__main__':
   # The ID of your GCS bucket goes here
  bucket_name = "NAME_HERE"

  for folder_name in ['train', 'test', 'checkpoints']:
    create_folder(bucket_name, folder_name)
```

### Uploading to GCP Buckets

Uploading a given shard and checkpoint to a GCP bucket is done with many helper functions. In order to direct each given shard to the appropriate dataset split, we first save the shard locally to the `data_dir` folder (which is included in our `.gitignore`).

```python
def upload_file(split):

  def upload_many_blobs_with_transfer_manager(split, filenames, source_directory="", workers=8):

    # split gives access to folders within GCP, ie "test/"
    blob_names = [split + name for name in filenames]

    # matches blob_name splits with their respective files in local memory
    blob_file_pairs = [(os.path.join(source_directory, f), bucket.blob(b)) for f, b in zip(filenames, blob_names)]

    # uploading the blob_file_pairs onto GCP, utilizes threading
    results = transfer_manager.upload_many(
      blob_file_pairs, skip_if_exists=True, max_workers=workers, worker_type=transfer_manager.THREAD
    )

  FILE_NAMES = os.listdir(DATA_CACHE_DIR)
  upload_many_blobs_with_transfer_manager(split, FILE_NAMES, DATA_CACHE_DIR, WORKERS)

  # cleanup
  for file in FILE_NAMES:
    full_path = DATA_CACHE_DIR + '/' + file
    os.remove(full_path)
```

## Shard Checkpointing

We introduce a method to checkpoint uploaded shards to the GCP bucket to avoid losing progress during tokenization, as the process often takes hours even on distributed systems. In our script, if passed the `--continue` argument, the script will look for the last uploaded checkpoint in the bucket's `checkpoints/` folder and use the HuggingFace datasets `.skip()` method to continue from the next required shard. This is done by keeping track of the number of documents processed in each checkpoint file alongside the document's unique ID as provided by FineWeb already.

```python
def upload_checkpoint():
  checkpoint_files = os.listdir(checkpoint_dir)
  for filename in checkpoint_files:
    blob = bucket.blob(f"checkpoints/{filename}")
    blob.upload_from_filename(os.path.join(checkpoint_dir, filename))
  for filename in checkpoint_files:
    os.remove(os.path.join(checkpoint_dir, filename))
```

The `upload_checkpoint` function checks the local checkpointing dir and simply scrapes its files to redirect them to the GCP bucket. This is akin to the data directory and each checkpointing file is fully self-contained in terms of the information we need to upload. The only caveat is that we need the latest shard as we sort by time when reading checkpoints (explained later). Thus, we make sure to remove all files in the local directory after upload, which ensures that only one checkpoint is present at a given time.

```python
fw.skip(skip_number)
print('total docs processed so far: ' + str(skip_number))
if continue_processing:
  print('skipped to the previous checkpoint')
```

The default value for `skip_number` is 0. The [datasets library](https://huggingface.co/docs/datasets/v1.11.0/dataset_streaming.html#split-your-dataset-with-take-and-skip) provides the `skip(n)` function which skips over the first `n` examples/rows in the given dataset. In the actual script, checkpointing is done only when it is ready to upload a shard. This ensures a guarantee that no previous progress has been lost, and the downtime for progress lost in between is little (<1 min for a single v4 TPU, <20s for newer versions TPU generations).

```python
with mp.Pool(nprocs) as pool:
  if continue_processing:
    shard_index = shard_to_resume + 1
  else:
    shard_index = 0
  ...
  for tokens, doc_id in pool.imap(tokenize, fw, chunksize=16):
    skip_number += 1
    if token_count + len(tokens) < shard_size:
      ...
    else:
      # checkpoint the shard
      checkpoint_filename = os.path.join(checkpoint_dir, f"{doc_id}.txt")
      with open(checkpoint_filename, "w") as f:
          f.write(str(shard_index) + ':' + str(skip_number))
      ...
      # upload file and checkpointing functions
      upload_file(split)
      upload_checkpoint()
```

Finally, in order to load the checkpoints at startup (only true if the `--continue` flag is provided), we check the GCP folder for the latest checkpoint sorted by time. Then, the `shard_to_resume` (number of shards already processed for future naming) and `skip_number` (number of document rows already processed) variables are pulled from the file data, which were determined at upload time.

```python
if continue_processing:
  # pull latest checkpoint name from gcp bucket called checkpoints
  blobs = bucket.list_blobs(prefix="checkpoints/")
  checkpoint_blobs = [b for b in blobs if str(b.name).endswith(".txt")]

  # if no checkpoints are found
  if not checkpoint_blobs:
    continue_processing = False
  else:
    latest_checkpoint = max(checkpoint_blobs, key=lambda b: b.updated)
    # grab shard id (in checkpoint name upon upload)
    checkpoint_to_resume = latest_checkpoint.name[len("checkpoints/"):-4]
    # parse file to get shard number and # of documents skipped
    shard_to_resume, skip_number = map(int, (latest_checkpoint.download_as_bytes().decode('utf-8')).split(':'))
```

## Distributed Multiprocessing using Ray

[Ray](https://docs.ray.io/en/latest/index.html) is an open-source framework for distributed machine learning applications. It provides an interface to connect multiple machines on the same network (for example, 32 v4 TPUs) into a "cluster" that can utilize all of the shared resources together.

{% include figure.liquid path="assets/img/tokenization/3.png" class="img-fluid" caption="Example Ray use-case to perform model parallelism (<a href='https://docs.ray.io/en/latest/ray-overview/use-cases.html'>source</a>)" %}

The Ray library exposes a multiprocessing API that is intended to directly replace Python's `multiprocessing` module. However before using Ray, we have to edit a few parts of the tokenization function so that it can support distributed operations. This is to ensure that data and information are shared correctly across different TPUs on different machines. For example, we are not able to use the `pool.imap` function anymore. This was better before as`pool.imap` streams results from workers incrementally (instead of waiting for all tasks like `map`), which saves memory and lets us shard, checkpoint, and upload on the go.

### Cluster Setup

Ray’s key idea is that any Python function can be turned into a remote task and run on any node in the cluster. You mark a function with `@ray.remote` decorator, call it with `.remote()`, and Ray takes care of process scheduling, inter-node communication, and result collection. Likewise, [cluster setup](https://docs.ray.io/en/latest/ray-core/starting-ray.html) is straightforward: one arbitrary machine acts as the head node, and others join as worker nodes. Once Ray is initialized, all nodes form a single logical pool of resources (CPUs, GPUs, memory). The following commands can also be found on the SSH startup script in our repo under the tokenization section.

```python
# on the head node
ray start --head --port=6379

# on each worker node (replace head-node-ip with what the prev. command outputs)
ray start --address='head-node-ip:6379'
```

After starting the cluster in the terminal, the follow code needs to be added to the tokenization file after downloading the required packages (guides found on [Ray docs](https://docs.ray.io/en/latest/cluster/vms/getting-started.html))

```python
import ray
ray.init(address="auto")  # connect to the cluster
```

### Adding Ray

In our original script, tokenization was done with `multiprocessing.Pool`. To move this to Ray, we convert the `tokenize()` function into a Ray task using the `ray.remote` decorator:

```python
@ray.remote
def tokenize(doc):
  doc_id = doc['id']
  tokens = [eot] # the special <|endoftext|> token delimits all documents
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
  tokens_np_uint32 = tokens_np.astype(np.uint32)
  return tokens_np_uint32, doc_id
```

Now, instead of running locally, each call to `tokenize_remote.remote(doc)` will be dispatched to any available worker across the cluster. Results are collected with `ray.get()`. Additionally, as we are no longer using `pool.imap()`, we have to create our own batches for the worker processes. This is done with a simple python list and a `while True` loop is added to maintain similar structure to previous script:

```python
while True:
  batch = []
  try:
    for _ in range(BATCH_SIZE):
      batch.append(next(doc_iter))
  except StopIteration:
    pass

  if not batch:
    break

  futures = [tokenize.remote(doc) for doc in batch]
  results = ray.get(futures)
```

`BATCH_SIZE` is a hyperparamater that must be optimized depending on the cluster configuration. The bash script for finding the optimal `BATCH_SIZE` value for your particular cluster is provided in the `tokenization/scripts/` folder.

### Final Script

With the cluster initialized, the tokenization function adapted for distributed execution, and batching logic added, we can now combine everything into the full Ray-enabled pipeline. The final script ties together streaming from Hugging Face, distributed tokenization across nodes, shard writing, checkpointing, and GCP uploads. Below is the complete version, with inline comments explaining each major step.

```python
import os
import shutil
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from google.cloud.storage import Client, transfer_manager
import argparse
import ray

# init ray in the cluster mode
ray.init(address="auto")

# constants for splits and multiprocessing
TEST_SPLIT = 350
BUCKET_NAME = "ray_jaxformer"
BATCH_SIZE = 512
WORKERS = int(os.cpu_count())
nprocs = max(1, int(os.cpu_count() / 1.5))

# other constants for dataset processing
local_dir = "data_dir"
remote_name = "sample-350BT"
shard_size = int(1e8)

# gcp storage client and bucket
storage_client = Client()
bucket = storage_client.bucket(BUCKET_NAME)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# set up argument parser to check if --continue flag is given
def setup_argument_parser():
  parser = argparse.ArgumentParser(description='Process the 350BT dataset')
  parser.add_argument('--continue', dest='continue_processing', action='store_true',
            help='Continue processing from a checkpoint')
  parser.set_defaults(continue_processing=False)
  return parser

parser = setup_argument_parser()
args = parser.parse_args()
continue_processing = args.continue_processing
checkpoint_to_resume = None
shard_to_resume = 0
skip_number = 0

# if a --continue flag is given, pull latest checkpoint name from gcp bucket called checkpoints
if continue_processing:
  # pull latest checkpoint name from gcp bucket called checkpoints
  blobs = bucket.list_blobs(prefix="checkpoints/")
  checkpoint_blobs = [b for b in blobs if str(b.name).endswith(".txt")]
  if not checkpoint_blobs:
    continue_processing = False
  else:
    latest_checkpoint = max(checkpoint_blobs, key=lambda b: b.updated)
    checkpoint_to_resume = latest_checkpoint.name[len("checkpoints/"):-4]  # remove 'checkpoints/' prefix and '.txt' suffix
    shard_to_resume, skip_number = map(int, (latest_checkpoint.download_as_bytes().decode('utf-8')).split(':'))

# ------------------------------------------

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# init the tokenizer
enc = tiktoken.encoding_for_model("gpt-4") # 'cl100k_base'
eot = enc._special_tokens['<|endoftext|>'] # end of text token

# tokenize function with ray remote decorator
@ray.remote
def tokenize(doc):
  doc_id_return = doc['id']
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
  tokens_np_uint32 = tokens_np.astype(np.uint32)
  return tokens_np_uint32, doc_id_return

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

# function to upload files to gcp bucket using transfer manager
def upload_file(split):
  def upload_many_blobs_with_transfer_manager(split, filenames, source_directory="", workers=8):

    blob_names = [split + name for name in filenames]

    blob_file_pairs = [(os.path.join(source_directory, f), bucket.blob(b)) for f, b in zip(filenames, blob_names)]

    results = transfer_manager.upload_many(
      blob_file_pairs, skip_if_exists=True, max_workers=workers, worker_type=transfer_manager.THREAD
    )

  FILE_NAMES = os.listdir(DATA_CACHE_DIR)
  upload_many_blobs_with_transfer_manager(split, FILE_NAMES, DATA_CACHE_DIR, WORKERS)
  for file in FILE_NAMES:
    full_path = DATA_CACHE_DIR + '/' + file
    os.remove(full_path)

# function to upload checkpoints to gcp bucket and remove local copies
def upload_checkpoint():
  checkpoint_files = os.listdir(checkpoint_dir)
  for filename in checkpoint_files:
    blob = bucket.blob(f"checkpoints/{filename}")
    blob.upload_from_filename(os.path.join(checkpoint_dir, filename))
  for filename in checkpoint_files:
    os.remove(os.path.join(checkpoint_dir, filename))

# skip to the previous checkpoint (zero by default)
fw.skip(skip_number)
shard_index = shard_to_resume + 1 if continue_processing else 0

# variables to keep track of tokens in the current shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
token_count = 0
progress_bar = None
doc_iter = iter(fw)

while True:
    batch = []
    try:
      for _ in range(BATCH_SIZE):
        batch.append(next(doc_iter))
    except StopIteration:
      pass

    if not batch:
      break

    # get the tokenized results from ray
    futures = [tokenize.remote(doc) for doc in batch]
    results = ray.get(futures)

    for tokens, doc_id in results:
      skip_number += 1

      # if the current document fits in the current shard
      if token_count + len(tokens) < shard_size:

        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)

        # update progress bar
        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}", dynamic_ncols=True)
        progress_bar.update(len(tokens))

      else:

        # save a checkpoint for resuming later
        checkpoint_filename = os.path.join(checkpoint_dir, f"{doc_id}.txt")
        with open(checkpoint_filename, "w") as f:
          f.write(str(shard_index) + ':' + str(skip_number))

        # write the current shard and start a new one
        if shard_index >= 0 and shard_index < TEST_SPLIT:
          split = 'test/'
          shard_index_number = shard_index
        else:
          split = 'train/'
          shard_index_number = shard_index - TEST_SPLIT
        split_name = split[:-1]

        filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")

        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

        write_datafile(filename, all_tokens_np)
        upload_file(split)
        upload_checkpoint()

        # update shard index and reset progress bar
        shard_index += 1
        progress_bar = None

        # populate the next shard with the leftovers of the current doc
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder

# write any remaining tokens as the last shard
if token_count != 0:
  if shard_index >= 0 and shard_index < TEST_SPLIT:
    split = 'test/'
    shard_index_number = shard_index
  else:
    split = 'train/'
    shard_index_number = shard_index - TEST_SPLIT
  split_name = split[:-1]

  filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")

  write_datafile(filename, all_tokens_np[:token_count])
  upload_file(split)
  upload_checkpoint()


# clean up directory after function terminates
if os.path.exists(DATA_CACHE_DIR):
  shutil.rmtree(DATA_CACHE_DIR)
```

Now that we are done with tokenization, we can move onto model architecture, starting with learning how to write the base model.
