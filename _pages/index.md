---
layout: distill
title: "Jaxformer"
subtitle: "A zero-to-one guide on scaling modern transformers with n-dimensional parallelism."
permalink: /
description: "NEED TO REPLACE THIS: Training LLMs often feels like alchemy, but understanding and optimizing the performance of your models doesn't have to. This book aims to demystify the science of scaling language models: how TPUs (and GPUs) work and how they communicate with each other, how LLMs run on real hardware, and how to parallelize your models during training and inference so they run efficiently at massive scale. If you've ever wondered “how expensive should this LLM be to train” or “how much memory do I need to serve this model myself” or “what's an AllGather”, we hope this will be useful to you."
date: 2025-09-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: false

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: ../tokenization
next_section_name: "Part 1: Tokenization"

bibliography: main.bib

citation: true

authors:
  - name: Aditya Makkar
    url: "https://www.linkedin.com/in/aditya-makkar-76a23a246/"
    affiliations: 
      name: University of Waterloo
  - name: Divya Makkar
    url: "https://www.linkedin.com/in/divya-makkar000/"
  - name: Chinmay Jindal
    url: "https://www.linkedin.com/in/chinmayjindal/"

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: High-Level Outline
  - name: Links to Sections

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

OK it starts here now i guess