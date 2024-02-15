# A Glitch in the Matrix? Locating and Detecting Language Model Grounding with Fakepedia

This repository contains the data and code to reproduce the results of our paper: https://arxiv.org/abs/2312.02073

Please use the following citation:

```
@misc{monea2023glitch,
      title={A Glitch in the Matrix? Locating and Detecting Language Model Grounding with Fakepedia}, 
      author={Giovanni Monea and Maxime Peyrard and Martin Josifoski and Vishrav Chaudhary and Jason Eisner and Emre Kıcıman and Hamid Palangi and Barun Patra and Robert West},
      year={2023},
      eprint={2312.02073},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

> **Abstract:** Large language models (LLMs) have an impressive ability to draw on novel information supplied in their context. Yet the mechanisms underlying this contextual grounding remain unknown, especially in situations where contextual information contradicts factual knowledge stored in the parameters, which LLMs also excel at recalling. Favoring the contextual information is critical for retrieval-augmented generation methods, which enrich the context with up-to-date information, hoping that grounding can rectify outdated or noisy stored knowledge. We present a novel method to study grounding abilities using Fakepedia, a dataset of counterfactual texts constructed to clash with a model's internal parametric knowledge. We benchmark various LLMs with Fakepedia and then we conduct a causal mediation analysis, based on our Masked Grouped Causal Tracing (MGCT), on LLM components when answering Fakepedia queries. Within this analysis, we identify distinct computational patterns between grounded and ungrounded responses. We finally demonstrate that distinguishing grounded from ungrounded responses is achievable through computational analysis alone. Our results, together with existing findings about factual recall mechanisms, provide a coherent narrative of how grounding and factual recall mechanisms interact within LLMs.