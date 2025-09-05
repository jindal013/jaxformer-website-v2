---
layout: distill
title: "Jaxformer"
subtitle: "Scaling Modern Transformers"
permalink: /
description: "This is a zero-to-one guide on scaling modern transformers with n-dimensional parallelism. Transformers have driven much of the deep learning revolution, yet no practical guide reflects SOTA architectures and the complexities of large-scale language modelling. While excellent resources such as DeepMind’s 'How to Scale Your Model' and HuggingFace’s 'Ultra Scale Playbook' exist, a gap remains between theory and end-to-end implementation. We aim to bridge that gap by showing you how to scale a model from scratch (in Jax, with code) to current standards."
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
  - name: Prerequisites

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

Find the complete code at the GitHub repository: [JAXformer](https://github.com/divyamakkar0/Jaxformer).

## Prerequisites

Prior to reading this guide, we assume you are famiilar with the following topics and resources (or equivalent material):

- Basic transformer implementations
- Basic familiarity with distributed training ideas
- JAX basics. Read through their [docs](https://docs.jax.dev/en/latest/)
- Andrej Karpathy's [Zero-to-Hero Neural Network](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series
