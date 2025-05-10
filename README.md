
# Optimizing BERT-base for Efficient News Classification

This project aims to optimize the **BERT-base** model for **efficient news topic classification**, incorporating techniques such as **LoRA** (Low-Rank Adaptation) and **mixed precision** to enhance classification accuracy and reduce computational overhead.

## Dataset

The dataset consists of **200,000+ news headlines** from the **HuffPost** spanning the years **2012-2018**. Each headline is associated with a corresponding **short description**. The task is to predict the category of news articles based on their headlines and descriptions. This model can be used to identify the category tags for untracked news articles and understand the type of language used across different news categories.

You can access the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

### Example Dataset Categories:

* Politics
* Business
* Entertainment
* Technology
* Science
* Health
* Sports

## Models Implemented

### BERT-base (Fine-Tuning)

The **BERT-base** model was fine-tuned on the dataset for news topic classification. Fine-tuning enables the model to adjust its weights for the specific task at hand, leveraging the pre-trained knowledge of BERT while tailoring it to the news categorization task.

### LoRA (Low-Rank Adaptation) for BERT

**LoRA** is integrated into the BERT model to enhance its performance. By reducing the computational burden of fine-tuning large models like BERT, LoRA allows us to adapt the pre-trained model with fewer parameters, improving efficiency.

### Mixed Precision Training

This approach is used to speed up the training process while maintaining the accuracy of the model. It reduces memory usage and speeds up the model's training without sacrificing model performance.

## Results

### Model Performance 
| Model                    | Test Accuracy | Test Loss   | Training Time (1 Epoch) |
|--------------------------|---------------|-------------|-------------------------|
| **BERT-base (Fine-Tuned)**| 71.25%        | 0.1288      | ~1600s                  |
| **BERT + LoRA + AMP**     | 61.78%        | 1.4186      | ~430s                   |

### Model Size
| Model                    | Trainable Parameters(K) | Model Size(MB)   | 
|--------------------------|---------------|-------------|
| **BERT-base (Fine-Tuned)**| 1110,000       | 419.03      | 
| **BERT + LoRA + AMP (Adapter Only)**     | 344.1       | 1.25      |

## Observation
**Latency**: LoRA + AMP reduced inference time from 21.44 ms → 14.64 ms, showing faster forward efficiency.
**Memory**: Memory usage in key ops (e.g., addmm, gelu) dropped by ~50%, indicating better GPU memory utilization.
**Training Efficiency**: Although validation accuracy dropped (↓ ~10%) due to fewer trainable parameters and FP16 precision, training time per epoch improved dramatically from 1600s to 430s.


## Requirements

To run this project, the following dependencies are required:

* **PyTorch 1.0 or higher** – For building and training the models
* **pytorch-pretrained-bert** – A library for using pre-trained BERT models ([https://github.com/huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT))
* **transformers** – For working with various transformer models ([https://huggingface.co/transformers/](https://huggingface.co/transformers/))


### Installation Instructions:

```bash
pip install torch
pip install transformers
pip install scikit-learn
pip install pytorch-pretrained-bert
```

## How to Run

1. Download the dataset from Kaggle and place it in the default folder.

2. Run each cell sequentially to:

* Preprocess the data

* Fine-tune the BERT model

* Evaluate the model performance
  
