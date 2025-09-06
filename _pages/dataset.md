---
layout: distill
title: "Dataset Class and Config Files"
permalink: /dataset/
description: "When training large-scale models on TPU or GPU clusters, memory-efficient data loading is needed to avoid bottlenecks. Below is a walkthrough of a custom Dataset class designed to stream and preprocess data shards from a Google Cloud Storage Bucket, supported for data, pipeline and tensor parallelism."
date: 2025-09-05
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: ../sharded
previous_section_name: "Part 3: Sharded Model"

next_section_url: ../distributed_training
next_section_name: "Part 5: Distributed Training"

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
  - name: The Dataset Class
  - name: Configs

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

## The Dataset Class

Beginning with the `Dataset` constructor, a `process_path` variable is declared as it will store the location of a shard's download from the GC Bucket.

```python
class Dataset
    def __init__(
        self,
        process_path : str,
        T: int,
        batch_size: int,
        microbatch: int,
        dp: int,
        pp: int,
        bucket_name: str,
        id: str,
        partition: Optional[NamedSharding] = None,
    ):
```

Then, the following assert statements are declared to ensure a reshaping can occur. For pipeline parallelism, the `batch_size` must divide into the `micro_batch` size and pipeline parallelism dimension must divide the `micro_batch` size.

```python
assert (batch_size % microbatch) == 0,
assert (microbatch % pp) == 0,
```

Other properties are also initialized, some noteworthy ones include `self.shard_idx`, `self.step_idx` and `self.id` which track the GCP shard to be streamed, the current training step index and the current data split's folder name (eg. `train`) respectively.

```python
class Dataset:
    def __init_(...):
        self.T = T
        self.batch_size = batch_size
        self.dp = dp
        self.microbatch = microbatch

        self.step_idx = 0
        self.shard_idx = 0
        self.partition = partition

        self.bucket_name = bucket_name
        self.base_process_path = process_path
        self.process_path = process_path
        self.id = id
        self.data = self.return_blobs(bucket_name, self.id)
        self.dir_name = "bucket_downloads"
        try:
            os.mkdir(self.dir_name)
        except OSError as e:
            print(f"{self.dir_name} already exists")
```

Another important instantiation is the `self.data` variable which holds a list of names containing the shards to be downloaded. The `bucket_name` and `self.id` (folder name) are taken as parameters and return a list containing all the names in the GCP bucket with the prefix identifier. Due to this, the folder name is also included which is why the first index in the resulting list is excluded.

```python
class Dataset:
    def __init__(...):
        ...

    def return_blobs(self, bucket_name, prefix, delimiter=None):
        res = []
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        for blob in blobs:
            res.append(blob.name)

        return res[1:]
```

Then, the process for downloading begins by calling the `load_next_shard()` function, which operates using the following 3 functions.

```python
class Dataset:
    def __init__(...):
        ...
        self.load_next_shard()
```

There are three functions that download a shard of data from the GCP bucket. The first is shown below and streams a file with a specific name from the GCP bucket.

```python
class Dataset:
    ...
    def download_blob_to_stream(self, bucket_name, source_blob_name, file_obj):
        """Downloads a blob to a stream or other file-like object."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_file(file_obj)
        print(f"Downloaded blob {source_blob_name} to file-like object.")

        return file_obj
```

The second function is wrapper around the first function. If the call to `download_blob_to_stream` is successful, then the result is returned, else the function is re-called after a 5 second wait.

```python
class Dataset:
    ...
    def download_bucket(self, bucket_name, source_name, f):
        while True:
            try:
                result = self.download_blob_to_stream(bucket_name, source_name, f)
                return result
            except Exception as e:
                log("Failed to download due to exception")
                time.sleep(5)
```

Note the `log` function is a simple way to ensure only one device is logging the download as stated by `jax.process_index() == 0`, instead of all devices printing the same message.

```python
def log(out: str):
    if jax.process_index() == 0:
        print(out)

class Dataset:
    ...
```

Lastly, `download_next` is the main function that executes the downloading. It creates a `source_name` by iterating through the `self.data` array with all the names of the files in the GCP bucket using the `shard_idx`. Then, a unique process path is created using the `shard_idx` and the file with `source_name` is downloaded.

```python
class Dataset:
    ...
    def download_next(self):
        log("Started downloading")
        source_name = self.data[self.shard_idx % len(self.data)]
        self.shard_idx += 1
        log(f" Downloading: {source_name} | Shard_idx: {self.shard_idx}")

        self.process_path = f"{self.base_process_path}_{self.id}_{self.shard_idx}"
        with open(self.process_path, "wb") as f:
            result = self.download_bucket(self.bucket_name, source_name, f)
            log(f"Done downloading {result}")
```

When the `load_next_shard()` function is called, it calls `self.download_next()` which was explained above. Once the shard has been downloaded, it must be processed - rearranged to accommodate the batch size and mini batch sizes for data/pipeline parallelism, and reshaped into the x and y datasets. This is done with the `process_prev` function which begins by using `np.load(self.process_path)` to load the `.npy` shard that was downloaded to the `self.process_path` to a numpy array called `data`. The features for the dataset are loaded started from the beginning of the data array, leaving out the last index. The labels start from the first index (note the data is 0-indexed) till the end of the array. The reason why the labels is shifted one value is due the nature of predicting the next token. For the 0th token of data, the next token to be predicted is the 1st index, hence the reason why the features stop at the `[:-1]` index as the last token is the predictor for the second last token.

```python
class Dataset:
    ...
    def load_next_shard(self):
        self.download_next()

        def process_prev():
            log(f"Processing shard at {self.process_path}\n\n")

            try:
                data = np.load(self.process_path)
            except:
                log(f"couldn't load data\n\n")
            self.dataset = data[:-1]
            self.labels = data[1:]

```

Now, at this stage, both the dataset and labels are reshaped to align with distributed training. The process begins by determining the total number of usable training samples(`len_dataset`) and calculating the maximum number of complete batches that can be formed. The dataset and corresponding labels are then trimmed and reshaped into a four-dimensional tensor of shape: $(\text{max\_batches},\; \text{microbatch},\; \tfrac{dp \times \text{batch\_size}}{\text{microbatch}},\; T)$ where `dp` is the number of data parallel instances, and `microbatch` is the number of microbatches per instance, the next term is the number of samples per microbatch and T is the sequence length. This structure ensures the data can be cleanly partitioned across multiple devices and supports microbatch based grad accumulation allowing for efficient JAX sharding and device transfer.

```python
def load_next_shard(self):
    ...
    def process_prev():
        ...
        len_dataset = self.dataset.shape[0]
        max_batches = len_dataset // (self.batch_size * self.T)

        self.dataset = self.dataset[:max_batches * self.batch_size * self.T * self.dp].reshape(
                max_batches,
                self.microbatch,
                (self.dp * self.batch_size) // self.microbatch,
                self.T,
            )
        self.labels = self.labels[
            : max_batches * self.batch_size * self.T * self.dp
        ].reshape(
            max_batches,
            self.microbatch,
            (self.dp * self.batch_size) // self.microbatch,
            self.T,
        )
```

In JAX, sharding refers to dividing an array across multiple devices, typically described using a `NamedSharding` object. This specifies how array dimensions should be partitioned across a device mesh (e.g., along data, pipeline, or tensor axes). In the code, the dataset and labels are placed on devices using `jax.device_put` with the given sharding specification. This ensures that each device receives only the portion of the data it is responsible for, rather than creating one large array and letting JAX scatter it afterward, saving memory and communication costs in the process. The process function is called, and the path is removed after.

```python
class Dataset:
    def load_next_shard(self):
        def process_prev():
            ...
            self.dataset = jax.device_put(self.dataset, self.partition)
            self.labels = jax.device_put(self.labels, self.partition)

        process_prev()

        os.remove(self.process_path)
```

Additionally, within the `Dataset` class, we have the length function which returns the number of batches available in the current loaded shard (0th dimension). Additionally, the `__call__` method is used to fetch the next batch of inputs and labels sequentially. The `step_idx` variable increments each call and if the index exceeds all the batches in the current shard, it means we have exceeded all the batches and we can reset the idx to 0 and load the next shard. A batch is extracted by slicing the dataset labels from `step_idx : step_idx + step`. This provides exactly step samples, which aids in the implementation of gradient accumulation. Finally, `step_idx` is incremented by `step`, so that the next call fetches the following batch. This creates a continuous stream of batches across shards.

```python
class Dataset:
    def __call__(self):
        if self.step_idx >= self.dataset.shape[0]:
            self.step_idx = 0
            self.load_next_shard()

        x = self.dataset[self.step_idx : self.step_idx + step]
        y = self.labels[self.step_idx : self.step_idx + step]
        self.step_idx += step

        return x, y
```

We can add a few more utility functions to create dataset from a dataset config as well as some properties listed below.

```python
class Dataset
    ...
    @classmethod
    def getDataset(
        cls,
        cfg: dataConfig,
        partition: Optional[NamedSharding] = None,
        dp: int = 1,
        pp: int = 1,
        tp: int = 1,
    ) -> Tuple["Dataset", "Dataset"]:
        assert (cfg.T % tp) == 0, "T should be divisible by tensor parallelism"
        train_dataset = cls(
            cfg.process_path,
            cfg.T,
            cfg.train_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.train_folder_name,
        )
        val_dataset = cls(
            cfg.process_path,
            cfg.T,
            cfg.val_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.val_folder_name,
        )

        return train_dataset, val_dataset

    def __len__(self):
        return self.dataset.shape[0]

    @property
    def tokens_per_step(self):
        return self.dp * self.batch_size * self.T
```

More advanced data loading techniques can be used such as disturbed data loading however we are able to bypass this and use this dataloader on a multi-node setting since the data chunks are in shards and thus it is still efficient for every process to download duplicate data.

## Configs

Here are the configs found in the `utils.py`. Beginning with the different config classes, they are configured for the model, dataset processing, learning rate/optimizer configs, device config for distributed training and inference config respectively.

```python
@dataclass
class modelConfig:
    """model config class"""

    model_dimension: int
    vocab_size: int
    n_head: int
    blocks: int
    layers_per_block: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: str = "bfloat16"

@dataclass
class dataConfig:
    bucket_name: str
    process_path: str = "./bucket_downloads/processShard"
    train_folder_name: str = "train"
    val_folder_name: str = "val"
    T: int = 6
    train_batch_size: int = 3
    val_batch_size: int = 3
    micro_batch_size: int = 1

@dataclass
class LRConfig:
    max_lr: float
    min_lr: float
    end_lr: float
    warmup_steps: int
    end_steps: int

@dataclass
class deviceConfig:
    n_device_axis: List[int]

@dataclass
class inferenceConfig:
    prompt: Optional[str] = None
    batch_size: int = 1
    top_k: int = 10000
    temperature: float = 1.0
    n_devices: int = 1
    max_tokens: int = 256
    use_cache: bool = True

@dataclass
class config:
    model_config: modelConfig
    data_config: dataConfig
    lr: LRConfig
    device_config: deviceConfig
    inference_config: inferenceConfig
    output_dir: str
    training_steps: int
    name: str
    grad_step: int = 1
    alpha: float = 0.001
    checkpoint_steps: int = 10
    eval_steps: int = 25
    seed: int = 0
    wandb: bool = True
    grad_clip_norm: float = 1.0
```

Then, the `parse_args()` function is designed to parse command line arguments in regards to the model call.

```python
def parse_args():
    parser = argparse.ArgumentParser(description="model training")
    parser.add_argument("--model_dimension", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers_per_block", type=int, default=3)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dhR", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--capacity_factor", type=float, default=1.5)
    parser.add_argument("--bucket_name", type=str, default="10bt_gpt2")
    parser.add_argument(
        "--process_path", type=str, default="./bucket_downloads/processShard")
    parser.add_argument("--train_folder_name", type=str, default="train")
    parser.add_argument("--val_folder_name", type=str, default="val")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--end_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--end_steps", type=int, default=19073)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--name", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="gs://results_jaxformer/")
    parser.add_argument("--checkpoint_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--grad_step", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--n_device_axis", type=int, nargs="*", default=[1])
    parser.add_argument("--inference_batch", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--prompt", type=str, default="hello world")
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
```

Then all the individual config classes are instantiated.

```python
def parse_args():
    ...
    model_cfg = modelConfig(
        model_dimension=args.model_dimension,
        vocab_size=args.vocab_size,
        n_head=args.n_head,
        blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        T=args.T,
        latent_dim=args.latent_dim,
        dhR=args.dhR,
        dropout_rate=args.dropout_rate,
        model_dtype=args.model_dtype,
        k=args.k,
        n_experts=args.n_experts,
        n_shared=args.n_shared,
        capacity_factor=args.capacity_factor,
    )

    data_cfg = dataConfig(
        bucket_name=args.bucket_name,
        process_path=args.process_path,
        train_folder_name=args.train_folder_name,
        val_folder_name=args.val_folder_name,
        T=args.T,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    lr_cfg = LRConfig(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        end_lr=args.end_lr,
        warmup_steps=args.warmup_steps,
        end_steps=args.end_steps,
    )

    device_cfg = deviceConfig(
        n_device_axis=args.n_device_axis,
    )

    inference_cfg = inferenceConfig(
        prompt=args.prompt,
        batch_size=args.inference_batch,
        top_k=args.top_k,
        temperature=args.temperature,
        n_devices=args.n_devices,
        max_tokens=args.max_tokens,
        use_cache=args.use_cache,
    )
```

Finally one wrapper `config` class containing all these instantiated subclasses is returned as the final config for the model.

```python
def parse_args():
    ...
    cfg = config(
        model_config=model_cfg,
        data_config=data_cfg,
        lr=lr_cfg,
        name=args.name,
        output_dir=args.output_dir,
        device_config=device_cfg,
        checkpoint_steps=args.checkpoint_steps,
        inference_config=inference_cfg,
        seed=args.seed,
        training_steps=args.training_steps,
        grad_step=args.grad_step,
        eval_steps=args.eval_steps,
        alpha=args.alpha,
        wandb=args.wandb,
        grad_clip_norm=args.grad_clip_norm,
    )

    return cfg
```

With this covered, we can now move on to one of the most fundamental topics of this guide: distributed training.
