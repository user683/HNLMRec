# Can LLM-Driven Hard Negative Sampling Empower Graph Collaborative Filtering? Findings and Potentials.

<p align="center">
<img src="Model_Pipeline.jpg" alt="RLMRec" />
</p>

This paper introduces the concept of Semantic Negative Sampling and explores how to optimize LLMs for 
high-quality, hard negative sampling. Specifically, we design an experimental pipeline that includes 
three main modules: profile generation, semantic negative sampling, and semantic alignment, to verify 
the potential of LLM-driven challenging negative sampling in enhancing the accuracy of collaborative 
filtering (CF). Further analysis reveals that this gap primarily arises from two major challenges: 
semantic shift and lack of behavioral constraints. To address these challenges, we propose a framework
called **HNLMRec**, which is based on fine-tuning LLMs supervised by collaborative signals.
Experimental results show that this framework outperforms traditional negative sampling and other
LLM-driven recommendation methods across multiple datasets, providing new solutions for empowering 
traditional RS with LLMs.

## 📝 Environment
```bash
pip install numba==0.53.1
pip intall numpy==1.20.3
pip install scipy==1.6.2
pip install torch>=1.7.0
```

## 📈 Dataset Information and Processing Details
In our paper, we conducted experiments using four datasets ( Toys & Games, CDs & Vinyl, 
Yelp2018, Amazon Electronics 2023), among which the Amazon Fashion dataset was primarily 
used to validate the generalization capability of the fine-tuned model on new datasets. 
The statistical information for all datasets is presented in the table below:
## 🔬 Model Training and Inference

## 👏 Acknowledgement