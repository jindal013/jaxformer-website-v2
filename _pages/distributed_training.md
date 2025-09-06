---
layout: distill
title: "Distributed Training: Scaling Transformers in Practice"
permalink: /distributed_training/
description: "We now introduce the main training script that will be used to launch the training. This covers the infrastructure, distributed functions and training loops that will sync all devices together."
date: 2025-09-05
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: ../dataset
previous_section_name: "Part 4: Dataset & Config"

next_section_url: ../moe
next_section_name: "Part 6: Mixture of Experts"

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
  - name: Setting Up the Distributed Environment
  - subsections:
      - name: "Configuring JAX & XLA Flags"
      - name: "Device Mesh Initialization"
  - name: Training Infrastructure
  - subsections:
      - name: "Checkpointing and State Management"
      - name: "Data Partitioning and Model Initialization"
  - name: Training and Evaluation Loops

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

## Setting Up the Distributed Environment

### Configuring JAX & XLA Flags

We will begin by configuring JAX. In JAX, XLA flags optimize performance and are related to communications that occur between GPUs . We can follow general practice to enable flags allowing for faster performance. Note we train on TPUs but the flags do not hurt performance in general.

```python
import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)
```

We can also use JAX's optional disk cache which enables JAX to store copies of complied programs on disk, saving recompilation time when running the same or similar tasks repeatedly. We use a remote file storage to sync cache across multi-controller nodes.

```python
jax.config.update("jax_compilation_cache_dir", "gs://jaxformer-cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
```

We can begin by writing a helper function to print values on multi-controller JAX since we only want to print once on the main node's process.

```python
def log(msg: str):
    if jax.process_index() == 0:
        print(msg)
```

### Device Mesh Initialization

We can now create a function to setup the mesh for the given TPU topology. The function takes in the number of devices per axes, the name of each axis and returns the `JAX` mesh. It first makes the devices into an np array and ensures we have the right number of devices in the desired mesh as the device count.

We try to use the `jax.make_mesh` function as that makes the most optimized mesh given the topology of TPUs; however, if it cannot, it throws an exception hence we wrap it in a try-catch and make the mesh ourself. The mesh is then returned.

```python
def init_devices(
    axes: Tuple[int, ...], axes_name: Tuple[str, ...]
) -> jax.sharding.Mesh:
    devices = np.array(jax.devices())
    # print for convenience
    # Assumes you are on TPU
    for idx in np.ndindex(devices.shape):
        d = devices[idx]
        log(
            f"  {idx} ID: {d.id}, Process: {d.process_index}, "
            f"Coords: {d.coords}, Core: {d.core_on_chip}"
        )

    assert devices.size == np.prod(axes), (
        f"Expected {np.prod(axes)} devices, got {devices.shape[0]}"
    )
    try:
        mesh = jax.make_mesh(axes, axes_name)
    except:
        log("Failed to create mesh with make_mesh, falling back to sharding.Mesh")
        mesh = jax.sharding.Mesh(devices.reshape(axes), axes_name)
    return mesh
```

## Training Infrastructure

With the helper functions established, we can now begin the `main` training loop function. Our main function will take in the `config` we described earlier.  Since we are assuming this script is for `3-D` parallelism, we can assign variables to the device size for each axis and setup the key with the initial seed.

```python
def main(cfg: config):
    key = jax.random.PRNGKey(cfg.seed)
    DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL = cfg.device_config.n_device_axis
```

We can now initialize and log our mesh.

```python
def main(cfg: config):
  ...
  axes = (*cfg.device_config.n_device_axis,)
  axes_name = ("dp", "pp", "tp")

  mesh = init_devices(axes, axes_name)
  log(mesh)
```

### Checkpointing and State Management

The next step is to setup checkpointing for our model using the `orbax` library and the `orbax.checkpointing` (ocp) module which conveniently handles checkpointing on multiprocess and remote storage for us. All we need to do is give it the google storage url and make sure to run it on every process. We first make the directory from the config by combining the GCS url with the unique name of the run and setup the orbax checkpoint manager. We can then use this to see if a latest step exists in which case we are loading from a previous run.

```python
checkpoint_dir = cfg.output_dir + cfg.name
options = ocp.CheckpointManagerOptions(max_to_keep=1)
checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
load = checkpoint_manager.latest_step() is not None
```

### Data Partitioning and Model Initialization

We begin setting up our dataset. We first make the data partition. Every data shard that is loaded will be of the form `(G, M , B, T)` where `G` is the total batches in a shard, `M` is the microbatches in the batch (for pipelining), `B` is the batch size per microbatch and `T` is the sequence length. Thus we want the `M` to be split amongst the pipeline, `B` to be split amongst the data and `T` to be spilt initially along `Tensor` as discussed previously. Thus we obtain the following `PartitionSpec` and `NamedSharding` and can use it to initialize our dataset class written previously.

```python
data_spec = P(None, "pp", "dp", "tp")
data_partition = jax.sharding.NamedSharding(mesh, data_spec)

train_dataset, val_dataset = Dataset.getDataset(
    cfg.data_config,
    partition=data_partition,
    dp=DATA_PARALLEL,
    pp=LAYER_PARALLEL,
    tp=TENSOR_PARALLEL,
)
```

We can now create our model using the `ShardedModel`, creating our init key and and initializing our params.

```python
model = shardedModel(cfg.model_config)

log("creating sharded model ...")
key, init_key = jax.random.split(key, 2)
params = model.init_weights(init_key, mesh)
```

We can now use these params to initialize our optimizer. Since the params are sharded, the optimizer states will be sharded as well.

```python
lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=cfg.lr.min_lr,
    peak_value=cfg.lr.max_lr,
    warmup_steps=cfg.lr.warmup_steps,
    decay_steps=cfg.lr.end_steps,
    end_value=cfg.lr.end_lr,
)

tx = optax.chain(
    optax.clip_by_global_norm(config.grad_clip_norm),
    optax.inject_hyperparams(optax.adamw)(learning_rate=lr_scheduler),
)
```

One bug that we observed in Optax is that the params with no dims (i.e scalar values) are not replicated across devices leading to errors when trying to reload from checkpoints and use them in distributed functions calls (i.e train step which is written below). Hence we can write a simple map function that says if the value has no dimensions, replicate it across each device.

```python
default_sharding = jax.sharding.NamedSharding(mesh, P())
opt_state = jax.tree.map(
    lambda x: x if jnp.ndim(x) != 0 else jax.device_put(x, default_sharding),
    tx.init(params),
)
```

We can now setup our misc variables such as our starting step, whether to use wandb (has to be enabled in config and process 0), and a placeholder for the id.

```python
init_step = 0
use_wandb = cfg.wandb is True and jax.process_index() == 0
wandb_id = None
```

We can also write a save-checkpoint function to ensure the PyTree saves. Note we decrement the `shard_idx` for the train/val dataset because when loading a shard, we increment by 1, so we want to revert that change.

```python
def make_save_tree(step):
  model_state = {
      "params": params,
      "opt_state": opt_state,
  }
  save_tree = {
      "state": model_state,
      "key": jax.device_get(key),
      "train_step_idx": train_dataset.step_idx,
      "train_shard_idx": (train_dataset.shard_idx - 1) % len(train_dataset.data),
      "val_step_idx": val_dataset.step_idx,
      "val_shard_idx": (val_dataset.shard_idx - 1) % len(val_dataset.data),
      "step": step,
  }
  metadata = {
      "wandb_id": wandb_id
  }
  return save_tree, metadata
```

Our `save_checkpoint` function can now just take the step and call the checkpoint manager.

```python
def save_checkpoint(
    step,
):
    save_tree, metadata = make_save_tree(step)
    checkpoint_manager.save(step, args=ocp.args.Composite(
      state=ocp.args.StandardSave(save_tree),
      metadata=ocp.args.JsonSave(metadata)
    ))
```

Before the main training functions or training loop, we should add model-loading logic if we want to resume from a checkpoint. Since we always initialize, we can pass Orbax the sharding and array metadata from the current parameters and use that to load with the correct sharding.

```python

def main(cfg: config):
    ...
    if load:

        # get PyTree metadata
        abstract_tree_map = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, make_save_tree(init_step)
        )

        # load checkpoint
        tree = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_tree_state),
                metadata=ocp.args.JsonRestore(),
        ))

        # assign all variables
        tree_state, tree_metadata = tree.state, tree.metadata

        init_step = tree_state["step"]
        log(f"loading checkpoint @ step {init_step}")

        key.key = tree_state["key"]
        params = tree_state["state"]["params"]
        opt_state = tree_state["state"]["opt_state"]

        train_dataset.step_idx = tree_state["train_step_idx"]
        train_dataset.shard_idx = tree_state["train_shard_idx"]
        train_dataset.load_next_shard()

        val_dataset.step_idx = tree_state["val_step_idx"]
        val_dataset.shard_idx = tree_state["val_shard_idx"]
        val_dataset.load_next_shard()

        wandb_id = tree_metadata["wandb_id"]
        if use_wandb:
            assert wandb_id is not None, "wandb_id is None"
            wandb.init(
                entity="waterloo2",
                project="jaxformer",
                name=cfg.name,
                resume="must",
                id=wandb_id,
                config=asdict(cfg),
            )
```

Otherwise, if we are not loading, we can save the first checkpoint and initialize the wandb run if needed.

```python
def main(cfg: config):
  ...
  if load:
      ...
  else:
      log("no checkpoint found, saving init copy")
      save_checkpoint(init_step)
      if use_wandb:
          wandb.init(
              entity="waterloo2",
              project="jaxformer",
              name=cfg.name,
              resume="allow",
              config=asdict(cfg),
          )
          wandb_id = wandb.run.id

```

Finally, we can print our parameter count for convenience.

```python
param_count = jax.tree.reduce(
    lambda x, y: x + y.size,
    params,
    0,
)
log(f"Total parameters: {param_count:,}")
```

## Training and Evaluation Loops

Now we can introduce the step functions that call our model. We begin by writing a general step that runs the model forward and returns the loss along with other metrics. Note that we will use communication operations (e.g., `pmean`), but since this will ultimately be wrapped under a `shard_map`, this is allowed (you cannot call `pmean` unless you are under a mesh context, as there is otherwise no information about the distributed setting). Our step function is defined by wrapping `loss_fn` in a closure under the training variable.

```python
def step(params, x, y, key, train):
  def loss_fn(params, x, y, key):
      ...
  return loss_fn(params, x, y, key)
```

We can first get the logits from the model by calling `pipe_step`, discarding the cache output.

```python
def loss_fn(...):
    logits, _ = model.pipe_step(
        params,
        x,
        key=key,
        train=train,
    )
```

We can first begin by stepping through. We can use the JAX built in function to turn logits into log-probs and reshape it into a 2D tensor combining all dims other then the distribution into a batch.

```python
def loss_fn(...):
    ...
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    M, B, T, V = logits.shape
    y = y.reshape(-1)
    log_probs = log_probs.reshape(M * B * T, V)
```

Note that logits is `4D` originally since it is a tensor with dimensions defined as microbatch, batches per microbatch, sequence and vocab. We can get our cross-entropy loss by applying a vmap over and selecting the index that using a dynamic slice method, negating and then meaning over it.

```python
def loss_fn(...):
    loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
    loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()
```

To perform FSDP, we average the loss over the `dp` axis. We do the same for the `pp` and `tp` axes as well, since batches span multiple devices. The `jax.grad` function will then handle the reverse communication operations needed to propagate the gradients.

```python
loss_cross = jax.lax.pmean(loss_cross, axis_name="dp")
loss_cross = jax.lax.pmean(loss_cross, axis_name="tp")
loss_cross = jax.lax.pmean(loss_cross, axis_name="pp")
```

We can make a dict of metrics and return that as well.  This will be useful when we have to log stats on MoE.

```python
metrics = {
    "loss": loss,
    "loss_cross": loss_cross,
}
return loss, metrics
```

Now we can get the partition spec of the params, model, key and write the distributed step function.

```python
param_spec = shardedModel.get_p_spec(
    [model.embedding, model.block], mesh, cfg.model_config
)
opt_spec = jax.tree.map(lambda x: x.sharding.spec, opt_state)
key_spec = P("dp", "pp", "tp")
```

Note each device gets a unique key since we want every operation done on every device to be unique otherwise there is no reason to use more then 1 device. We start by writing the train step. In here, our step function will be the value and grad of step functions previously written since we also want to compute the gradients.

```python
def train_step(params, opt_state, x, y, key):
    step_fn = jax.value_and_grad(step, has_aux=True)
```

Then to allow for gradient accumulation we write a single step function. We take in the past gradients and the batch and then accumulate the grads.

```python
def train_step(params, opt_state, x, y, key):
    step_fn = jax.value_and_grad(step, has_aux=True)

    def single_step(grads, batch):
        (_, metrics), grads_current = step_fn(params, *batch, train=True)
        grads = jax.tree.map(lambda x, y: x + y, grads, grads_current)
        return grads, metrics
```

We can then initialize the grads and reshape the keys to be PRNG keys again (get rid of leading dims) and then use the `jax.lax.scan` function to sequentially loop over the leading dim of the `(x,y,key)` batch.

```python
def train_step(params, opt_state, x, y, key):
    step_fn = jax.value_and_grad(step, has_aux=True)

    def single_step(grads, batch):
        ...
        return grads, metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    key = key.reshape(cfg.grad_step, 2)
    grads, metrics = jax.lax.scan(
        single_step,
        grads,
        (x, y, key),
  )

```

We then average the gradients and metrics, apply the updates to the parameters, and return the updated parameters, optimizer state, and metrics. Thus, our final function is as follows:

```python
def train_step(params, opt_state, x, y, key):
    step_fn = jax.value_and_grad(step, has_aux=True)

    def single_step(grads, batch):
        (_, metrics), grads_current = step_fn(params, *batch, train=True)
        grads = jax.tree.map(lambda x, y: x + y, grads, grads_current)
        return grads, metrics

    grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    key = key.reshape(cfg.grad_step, 2)

    grads, metrics = jax.lax.scan(
        single_step,
        grads,
        (x, y, key),
    )

    grads = jax.tree.map(lambda x: x / cfg.grad_step, grads)
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)

    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, metrics
```

We now can wrap this function under a shard map to allow for the distributed training to occur. For the arguments, we use the spec we have defined throughout the script and the outputs follow the same way. Metrics are replicated across every device.

```python
@jax.jit
@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(param_spec, opt_spec, data_spec, data_spec, key_spec),
    out_specs=(param_spec, opt_spec, P()),
    check_vma=False,
)
def train_step(params, opt_state, x, y, key):
    ...
    return params, opt_state, metrics
```

We can similarly write the `eval_step` function. The only difference is we don't have to keep the grads, thus the carry argument of the `jax.lax.scan` can be ignored in the `single_step` function.

```python
@jax.jit
@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(param_spec, data_spec, data_spec),
    out_specs=P(),
    check_vma=False,
)
def eval_step(params, x, y):
    def single_step(_, batch):
        loss, metrics = step(
          params, *batch, key=jax.random.PRNGKey(0), train=False
        )  # Key does not matter
        return loss, metrics

    _, metrics = jax.lax.scan(single_step, 0, (x, y))
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)
    return metrics
```

We now define our final variables and sync devices. Note we split the sample key before the loop since we want to have the same random key for each inference to see how the model evolves. We also keep an array to append the training loss and average when we need to print for each eval step.

```python
def main(cfg: config):
    ...
    total_steps = cfg.training_steps
    total_tokens = train_dataset.tokens_per_step

    jax.experimental.multihost_utils.sync_global_devices("sync")
    log(f"Total steps: {total_steps}")
    log(f"Total tokens per step: {total_tokens:,}")

    key, sample_key = jax.random.split(key, 2)
    start = time.time()
    train_loss = [] # used to keep track of loss and averaged when printing
```

Our last helper function is to make the keys. Essentially we want to create new keys for each device for a total of our grad steps. Therefore, we can make this a param and re-jit the function for each new value since it must be a static parameter. This doesn't slow us down since this function is called and compiled once.

```python
@partial(jax.jit, static_argnames=["steps"])
def make_sharded_key(key, steps=1):
    key = jax.random.split(
      key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL * steps
    ) # python array currently make it into a jax array
    key = jnp.asarray(key).reshape(
      (DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, steps, 2)
    )
    return key
```

Then, the final training loop can be written. We start by splitting our key and then making our train keys, getting our data and finally calling the train step.

```python
def main(cfg: config):
    for current_step in range(init_step, total_steps):
        key, train_key = jax.random.split(key)
        train_key = make_sharded_key(train_key, steps=cfg.grad_step)

        x, y = train_dataset(step=cfg.grad_step)

        params, opt_state, metrics = train_step(params, opt_state, x, y, train_key)
        train_loss.append(metrics["loss"])
```

We then add wandb logging metrics and add in our eval step.

```python

def main(cfg: config):
    for current_step in range(init_step, total_steps):
        ...
        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/train_loss": metrics["loss"],
                "loss/train_cross_entropy_loss": metrics["loss_cross"],
                "lr": opt_state[1].hyperparams["learning_rate"],
            }

            if current_step % cfg.checkpoint_steps == 0:
                time_per_batch = time.time() - start
                eval_x, eval_y = val_dataset(step=cfg.eval_steps)
                val_metrics = eval_step(params, eval_x, eval_y)

                if use_wandb:
                    wandb_log["loss/val_loss"] = val_metrics["loss"]
                    wandb_log["loss/val_cross_entropy_loss"] = val_metrics["loss_cross"]

                jax.experimental.multihost_utils.sync_global_devices("sync")

                tokens_per_second = cfg.checkpoint_steps * total_tokens / time_per_batch
                train_loss = jnp.array(train_loss).mean().item()
                eval_loss = val_metrics["loss"].item()
                log_string = f"Step {current_step + 1}, Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
                log(log_string)
```

To avoid slowdown we can checkpoint every 10 eval steps. We can also include checkpointing to get the final training loop.

```python
def main(cfg: config):
    ...
    for current_step in range(init_step, total_steps):
        key, train_key = jax.random.split(key)
        train_key = make_sharded_key(train_key, steps=cfg.grad_step)

        x, y = train_dataset(step=cfg.grad_step)

        params, opt_state, metrics = train_step(params, opt_state, x, y, train_key)
        train_loss.append(metrics["loss"])

        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/train_loss": metrics["loss"],
                "loss/train_cross_entropy_loss": metrics["loss_cross"],
                "lr": opt_state[1].hyperparams["learning_rate"],
            }

        if current_step % cfg.checkpoint_steps == 0:
            time_per_batch = time.time() - start
            eval_x, eval_y = val_dataset(step=cfg.eval_steps)
            val_metrics = eval_step(params, eval_x, eval_y)

            if use_wandb:
                wandb_log["loss/val_loss"] = val_metrics["loss"]
                wandb_log["loss/val_cross_entropy_loss"] = val_metrics["loss_cross"]

            jax.experimental.multihost_utils.sync_global_devices("sync")

            tokens_per_second = cfg.checkpoint_steps * total_tokens / time_per_batch
            train_loss = jnp.array(train_loss).mean().item()
            eval_loss = val_metrics["loss"].item()
            log_string = f"Step {current_step + 1}, Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
            log(log_string)
            start = time.time()
            train_loss = []

    if current_step % 10 * cfg.checkpoint_steps == 0:
        save_checkpoint(current_step)
        if use_wandb:
            wandb.log(data=wandb_log, step=current_step)
```

Finally, we end the main function by calling `wandb.finish()` if we are using wandb. To kick off training, we can add a main guard that called `jax.distrbuted.intialize()` to sync the multi-controller processes and print the `cfg` from the `parse_args()`.

```python
if __name__ == "__main__":
    jax.distributed.initialize()
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))
    main(cfg)
```

We now look at how to scale this model further with MoE.
