---
layout: distill
title: "Conclusion: Summary and Future Extensions"
permalink: /conclusion/
description: ""
date: 2025-09-06
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: ../training
previous_section_name: "Part 7: Training Results"

next_section_url:
next_section_name: "The End"

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
  - name: Conclusion
  - name: Article Summaries
  - name: Future Directions
  - name: Authors & Contact
  - name: Links
#   - subsections:
#       - name: "Visualizing rooflines"
#       - name: "Matrix multiplication"
#       - name: "Network communication rooflines"
#   - name: A Few Problems to Work

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

## Conclusion

Throughout this guide, we walked step-by-step through building a modern, scalable transformer model in JAX, focusing on both architectural advances and scaling practices.

### Article Summaries

- Part 1: Tokenization — Efficient large-scale tokenization pipelines with sharding, checkpointing, and distributed uploads.
- Part 2: Base Model — Implementation of a single-GPU transformer with modern modules (RMSNorm, RoPE, MLA, KV-cache).
- Part 3: Sharded Model — A deep dive into 3-D parallelism (data, pipeline, and tensor) with practical JAX code.
- Part 4: Dataset & Config — Preparing datasets, configs, and orchestration for large-scale training.
- Part 5: Distributed Training - Scaling across nodes with Cloud TPU clusters.
- Part 6: Mixture of Experts - Implementing MoE layers (DeepSeek-style) with load balancing, stability tricks, and parallelism

Together, these parts form a zero-to-one guide on how to scale transformers from a simple baseline to cutting-edge distributed training.

### Future Directions

In the future, this can be extended further by using more novel methods such as replacing GPipe with DualPipe and incorporating higher dimensions of parallelism such as expert, and/or sequence. We can also extend the tokenziation process by streaming Parquet files over a distributed network.

**\*How to get in touch:** leave a comment on any page, reach us on socials, or start a discussion thread on the Github repo.

### Authors & Contact

We are all currently 1st and 2nd year undergraduate students at the University of Waterloo studying Computer Science.

| Author             | Twitter / X                                       | LinkedIn                                                              |
| ------------------ | ------------------------------------------------- | --------------------------------------------------------------------- |
| **Aditya Makkar**  | [@AdityaMakkar000](https://x.com/AdityaMakkar000) | [Aditya Makkar](https://www.linkedin.com/in/aditya-makkar-76a23a246/) |
| **Divya Makkar**   | [@\_DivyaMakkar](https://x.com/_DivyaMakkar)      | [Divya Makkar](https://www.linkedin.com/in/divya-makkar000/)          |
| **Chinmay Jindal** | [@chinmayjindal\_](https://x.com/chinmayjindal_)  | [Chinmay Jindal](https://www.linkedin.com/in/chinmayjindal/)          |

### Links

- Star the GitHub Repository (model): [JAXformer](https://github.com/divyamakkar0/Jaxformer)
