# Text Classification with LSTM

## Overview
This project classifies IMDB movie reviews as **Positive** or **Negative** using a Long Short-Term Memory (LSTM) network.

## Dataset
The LSTM uses the **same dataset, train-test split, tokenization, tensors, and labels** from the RNN activity.

- Dataset: IMDB Dataset of 50K Movie Reviews
- File: `IMDB Dataset.csv`
- 95% of the available data was used.

## Implementation

1. **Reuse Activity 1 data**
   - Reuse the same train/test split.
   - Reuse the same tokenized and padded tensors.

2. **Tokenization**
   - `tiktoken` with `cl100k_base` was used in Activity 1.

3. **Padding and truncation**
   - Maximum sequence length: **200 tokens**.

4. **LSTM model**

   `Embedding → LSTM → Dropout → Linear`

   Model settings:
   - Embedding size: 128
   - Hidden size: 128
   - 2 LSTM layers
   - Bidirectional LSTM
   - Dropout: 0.4

5. **Training**
   - Loss: Cross Entropy Loss
   - Optimizer: AdamW
   - Learning rate: 0.001
   - Gradient clipping
   - Maximum epochs: 8

6. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

7. **Custom reviews**
   - The same three reviews from Activity 1 were tested.

## Results

- Accuracy: **86.76%**
- Precision: **86.28%**
- Recall: **87.59%**
- F1 Score: **86.93%**

## RNN vs LSTM

| Metric | RNN | LSTM |
|---|---:|---:|
| Accuracy | 62.22% | 86.76% |
| Precision | 62.79% | 86.28% |
| Recall | 61.10% | 87.59% |
| F1 Score | 61.93% | 86.93% |

## Custom Review Results

The RNN and LSTM agreed on the first two reviews.

For the third review:
- RNN → Negative
- LSTM → Positive

The LSTM prediction was more reasonable because the review contained clearly positive wording.

## Conclusion
The LSTM performed considerably better than the RNN. Its memory and gating mechanism helped it learn sentiment patterns more effectively in the movie reviews.
