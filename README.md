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

|         Dataset         | #Users | #Items | #Interactions |  Density   |
|:-----------------------:|:------:|:------:|:-------------:|:----------:|
|      Toys & Games       | 22,338 | 9,023  |    200,511    | 1.0 × 10⁻³ |
|       CDs & Vinyl       | 19,385 | 8,279  |    186,535    | 1.2 × 10⁻³ |
|        Yelp2018         | 29,832 | 16,781 |    513,976    | 1.0 × 10⁻³ |
| Amazon Electronics 2023 | 97,570 | 44,669 |    178,259    | 4.1 × 10⁻⁷ |

For the Toys & Games dataset, we first filtered out interactions with ratings below 4 
and selected records within the date range from 2015-01-01 to 2018-01-01. Additionally,
we retained only users and items with at least 10 interactions. For the CDs & Vinyl 
dataset, we selected records within the date range from 2014-01-01 to 2016-01-01, with
the remaining processing steps consistent with the above. For the Yelp2018 dataset, 
we selected records within the date range from 2015-01-01 to 2018-01-01, with the
remaining processing steps also consistent with the above. As for the Amazon 
Electronics 2023 dataset, we retained only interactions with ratings of 3 or higher,
within the date range from 2010-01-01 to 2018-01-01.

## 🔬 Model Training and Inference

The pipeline of our model primarily consists of three parts: user-item profile generation, 
semantic negative sampling, and semantic alignment.

### User Profile and Item Profile Generation

```
{ 
  
}
```


## 👏 Acknowledgement