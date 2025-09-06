---
layout: distill
title: "Jaxformer"
subtitle: "Scaling Modern Transformers"
permalink: /
description: "This is a zero-to-one guide on scaling modern transformers with n-dimensional parallelism. Transformers have driven much of the deep learning revolution, yet no practical guide reflects SOTA architectures and the complexities of large-scale language modelling. While excellent resources such as DeepMind’s <a href='https://jax-ml.github.io/scaling-book/' target='_blank'>How to Scale Your Model</a> and HuggingFace’s <a href='https://huggingface.co/spaces/nanotron/ultrascale-playbook' target='_blank'>Ultra Scale Playbook</a> exist, a gap remains between theory and end-to-end implementation. We aim to bridge that gap by showing you how to scale a model from scratch (in Jax, with code) to current standards."
date: 2025-09-05
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: ../tokenization
next_section_name: "Part 1: Tokenization"

bibliography: main.bib

citation: true

authors:
  - name: Aditya Makkar
    url: "https://x.com/AdityaMakkar000"
  - name: Divya Makkar
    url: "https://x.com/_DivyaMakkar"
  - name: Chinmay Jindal
    url: "https://x.com/chinmayjindal_"

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Code & Contact
  - name: Introduction
  - name: Prerequisites
  - name: Goals
  - name: Overview

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

<img id="banner"
     class="img-fluid"
     alt="Banner"
     src="{{ 'assets/img/banner-light.png' | relative_url }}"
     data-light-src="{{ 'assets/img/banner.png' | relative_url }}"
     data-dark-src="{{ 'assets/img/banner-light.png' | relative_url }}" />

<script>
  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('light-toggle');
    const banner = document.getElementById('banner');

    function swapBanner() {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      banner.src = isDark ? banner.dataset.darkSrc : banner.dataset.lightSrc;
    }

    swapBanner();

    btn.addEventListener('click', () => requestAnimationFrame(swapBanner));
  });
</script>

## Code & Contact

Find the complete code for this guide on our [GitHub repository](https://github.com/divyamakkar0/Jaxformer).
More information about the authors can be found in the [Conclusion](https://jaxformer.com/conclusion).

## Introduction

Modern transformers are at the heart of today's deep learning systems, but taking them from a single-GPU prototype to a multi-node cluster is not straightforward. Scaling efficiently requires understanding how data moves through the hardware, how models can be split across devices, and how training infrastructure ties everything together.

This guide is a practical, code-first walkthrough of scaling modern transformers in JAX. Our goal is to bridge the gap between high-level scaling theory and hands-on implementation. By the end, you should feel comfortable building a SOTA transformer model that runs on TPUs/GPUs, sharding it across devices, and training it at scale with techniques used in SOTA systems.

## Prerequisites

Prior to reading this guide, we assume you are famiilar with the following topics and resources (or equivalent material):

- Basic Transformer implementations
- Familiarity with Distributed Training ideas
- JAX basics: we recommend that you start reading through their [docs](https://docs.jax.dev/en/latest/)
- Andrej Karpathy's [Zero-to-Hero Neural Network](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series

## Goals

By the end of this guide, you should be able to:

- Understand how to tokenize and stream large datasets efficiently for training.
- Estimate the compute, memory, and communication costs of running a transformer model.
- Select and combine parallelism schemes (data, tensor, pipeline, FSDP, MoE) for a given hardware setup.
- Confidently configure and launch distributed training runs on multi-host TPU or GPU clusters.
- Recognize bottlenecks that prevent strong scaling and know how to address them.

This is v1.0. We aim to update the guide sporadically as we implement more complex ideas and architectures in the future.

## Overview

Here's how the guide is structured:

- **[Part 1: Tokenization at Scale](tokenization)** — how to preprocess massive datasets, shard them, and checkpoint safely for distributed training.
- **[Part 2: Base Model](base_model)** — building a transformer in JAX with modules like RMSNorm, RoPE, and Multi-latent Attention.
- **[Part 3: Sharded Model](sharded)** — introducing parallelism strategies (data, tensor, pipeline, FSDP) and applying them to transformer layers.
- **[Part 4: Distributed Training](distributed_training)** — how to set up TPU/GPU clusters, manage checkpoints, and synchronize training loops.
- **[Part 5: Dataset & Configs](dataset_class)** — structured configs for datasets, hyperparameters, and runtime options.
- **[Part 6: Mixture of Experts](moe)** — implementing and training MoE layers, covering routing, stability, and efficiency challenges.
- **[Part 7: Final Run](final_run)** — putting it all together: multi-host scripts, launching large runs across TPU pods, and analyzing results.
- **[Part 8: Conclusions](conclusion)** — lessons learned, future directions like DualPipe and expert parallelism, and additional resources.
