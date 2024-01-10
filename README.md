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

> **Abstract:** Large language models (LLMs) have demonstrated impressive capabilities in storing and recalling factual knowledge, but also in adapting to novel in-context information. Yet, the mechanisms underlying their in-context grounding remain unknown, especially in situations where in-context information contradicts factual knowledge embedded in the parameters. This is critical for retrieval-augmented generation methods, which enrich the context with up-to-date information, hoping that grounding can rectify the outdated parametric knowledge. In this study, we introduce Fakepedia, a counterfactual dataset designed to evaluate grounding abilities when the parametric knowledge clashes with the in-context information. We benchmark various LLMs with Fakepedia and discover that GPT-4-turbo has a strong preference for its parametric knowledge. Mistral-7B, on the contrary, is the model that most robustly chooses the grounded answer. Then, we conduct causal mediation analysis on LLM components when answering Fakepedia queries. We demonstrate that inspection of the computational graph alone can predict LLM grounding with 92.8% accuracy, especially because few MLPs in the Transformer can predict non-grounded behavior. Our results, together with existing findings about factual recall mechanisms, provide a coherent narrative of how grounding and factual recall mechanisms interact within LLMs. 