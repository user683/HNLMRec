# Can LLM-Driven Hard Negative Sampling Empower Graph Collaborative Filtering? Findings and Potentials.

<p align="center">
<img src="Model_Pipeline.jpg" alt="RLMRec" />
</p>

Hard negative samples can accelerate model convergence and optimize decision boundaries, which is key 
to improving the performance of recommendation systems. Although large language models (LLMs) possess 
strong semantic understanding and generation capabilities, systematic research has not yet been conducted 
on how to effectively generate hard negative samples. To fill this gap, this paper introduces the concept 
of Semantic Negative Sampling and explores how to optimize LLMs for high-quality, hard negative sampling.
Specifically, we design an experimental pipeline that includes three main modules: profile generation,
semantic negative sampling, and semantic alignment, to verify the potential of LLM-driven challenging 
negative sampling in enhancing the accuracy of collaborative filtering (CF). Experimental results 
indicate that challenging negative samples generated based on LLMs, when semantically aligned and integrated 
into CF, can significantly improve CF performance, although there is still a certain gap compared to 
traditional negative sampling methods. Further analysis reveals that this gap primarily arises from
two major challenges: semantic shift and lack of behavioral constraints. To address these challenges, 
we propose a framework called **HNLMRec**, which is based on fine-tuning LLMs supervised by collaborative 
signals. Experimental results show that this framework outperforms traditional negative sampling
and other LLM-driven recommendation methods across multiple datasets, providing new solutions for 
empowering traditional RS with LLMs. Additionally, we validate the excellent generalization ability
of the LLM-based semantic negative sampling method on new datasets, demonstrating its potential in
alleviating issues such as data sparsity, popularity bias, and the problem of false hard negative samples.