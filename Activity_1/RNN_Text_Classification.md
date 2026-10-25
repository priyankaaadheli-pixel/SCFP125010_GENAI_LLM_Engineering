# Text Classification with RNN

## Overview
This project classifies IMDB movie reviews as **Positive** or **Negative** using a Recurrent Neural Network (RNN).

## Dataset
- Dataset: IMDB Dataset of 50K Movie Reviews
- File: `IMDB Dataset.csv`
- 95% of the available data was used.

## Implementation

1. **Load and clean data**
   - Load the CSV using Pandas.
   - Remove missing and duplicate reviews.

2. **Convert labels**
   - `negative` → `0`
   - `positive` → `1`

3. **Train-test split**
   - Use a stratified split to maintain the class distribution.

4. **Tokenization**
   - Use `tiktoken` with `cl100k_base`.

5. **Padding and truncation**
   - Set the maximum sequence length to **200 tokens**.
   - Longer reviews are truncated and shorter reviews are padded.

6. **Convert to tensors**
   - Reviews and labels are converted to PyTorch tensors.

7. **RNN model**

   `Embedding → RNN → Dropout → Linear`

   Model settings:
   - Embedding size: 128
   - Hidden size: 128
   - 2 RNN layers
   - Bidirectional RNN
   - Dropout: 0.4

8. **Training**
   - Loss: Cross Entropy Loss
   - Optimizer: AdamW
   - Learning rate: 0.001
   - Gradient clipping
   - Maximum epochs: 8

9. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

10. **Custom reviews**
    - Three new movie reviews were tested.

## Results

- Accuracy: **62.22%**
- Precision: **62.79%**
- Recall: **61.10%**
- F1 Score: **61.93%**

## Conclusion
The RNN successfully classified movie reviews, but its performance was lower than the LSTM model used in the second activity.
