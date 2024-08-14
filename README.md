# Multi-Modal Multi-Focus Emotion Recognition

This repository contains the code for processing and analyzing the IEMOCAP (Interactive Emotional Dyadic Motion Capture Database) dataset for emotion recognition from speech. The dataset contains audio files of emotionally expressive speech and corresponding metadata. The code in this repository processes the metadata, extracts features from the audio files, and prepares the dataset for further analysis.

## Overview of our approach
1. Acoustic frame-level features and lexical word embeddings are used as input for separate BLSTMs.
2. A context-based attention module is applied to pool the outputs of the BLSTMs and generate an utterance-level temporal aggregate.
3. The temporal aggregates from the two models are fused using an attention-based modality fusion module.
4. The fused output is passed through a linear softmax layer to get the classification probabilities.

## Dataset

The IEMOCAP dataset is not included in this repository. You need to obtain the dataset from the [official website](https://sail.usc.edu/iemocap/). After obtaining the dataset, unzip it and place it in the root directory of the project.

## Downloading the Large Data File

The data file `data_processed.pkl` (670MB) is too large to be uploaded to GitHub. This file is created after running the `DataProcessing.ipynb` notebook (on the IEMOCAP dataset), which preprocesses the data and extracts features. You can either run the notebook to create the pickle file or download it from Google Drive using the following link:

[Download data_processed.pkl from Google Drive](<https://drive.google.com/file/d/18lCvVF1T9yE884QmQtFwlw7E8id9DlBY/view?usp=share_link>)

## Usage

Once you've either created or downloaded the `data_processed.pkl` file and placed it in your project directory, you can load the data using the following code:

```python
import pandas as pd

# Load the DataFrame from the pickle file
data = pd.read_pickle("data_processed.pkl")
```

## Notebooks

### 1. DataProcessing.ipynb

Description:
- Explore dataset structure.
- Process metadata and transcripts.
- Refine emotion categories.
- Extract LLD features with OpenSMILE.
- Remove samples with non-text characters.

Results:
- The final dataset will be stored in the data DataFrame and is ready for further processing and analysis.

### 2. UnimodalClassifier.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

### 3. MMClassifier.ipynb

Description:
- [Add the description of the notebook here]

##Results:

### Results for Acoustic only models:
**Train Results:**

| Model | Avg Loss | Weighted Accuracy | Unweighted Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|-------|-----------|------------------|---------------------|------------|------------|--------------|----------|
| Last Block | 1.1084 | 49.95% | 51.57% | 52.19% | 37.83% | 48.80% | 67.46% |
| Avg. Pool  | 0.9579 | 61.41% | 62.69% | 65.69% | 51.56% | 60.72% | 72.79% |
| Attention Weighted Pooling| 0.8352 | 65.31% | 66.54% | 70.75% | 57.51% | 62.97% | 74.94% | 


**Test Results:**

| Model | Avg Loss | Weighted Accuracy | Unweighted Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|-------|-----------|------------------|---------------------|------------|------------|--------------|----------|
| Last Block  | 1.1296 | 49.82% | 50.03% | 38.79% | 38.44% | 57.91% | 64.97% |
| Avg. Pool | 1.0815 | 54.98% | 54.45% | 37.38% | 49.38% | 63.00% | 68.02% |
| Attention Weighted Pooling| 0.959 | 60.05% | 60.89% | 61.21% | 46.88% | 65.42% | 70.05% | 

### **Results for Lexical Only:**
**Train Results:**

| Model | Avg Loss | Weighted Accuracy | Unweighted Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|-------|-----------|------------------|---------------------|------------|------------|--------------|----------|
| Last Block | 0.7415 | 71.04% | 71.17% | 71.88% | 73.30% | 67.84% | 71.66% |
| CLS Block | 0.6517 | 75.52% | 75.71% | 81.44% | 76.89% | 72.64% | 71.88% |
| Avg. Pool | 0.7926 | 70.70% | 70.29% | 71.65% | 68.73% | 75.71% | 65.08% |
| Attention Weighted Pooling | 0.6387 | 75.02% | 74.96% | 76.49% | 74.98% | 75.56% | 72.79% |

**Test Results:**

| Model | Avg Loss | Weighted Accuracy | Unweighted Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|-------|-----------|------------------|---------------------|------------|------------|--------------|----------|
| Last Block | 0.9051 | 63.04% | 61.50% | 56.54% | 68.12% | 67.02% | 54.31% |
| CLS Block | 0.9369 | 63.13% | 63.18% | 62.62% | 65.00% | 61.66% | 63.45% |
| Avg. Pool| 1.0551 | 62.68% | 60.56% | 52.34% | 70.31% | 67.83% | 51.78% |
| Attention Weighted Pooling| 0.9641 | 63.13% | 64.83% | 73.36% | 55.62% | 59.79% | 70.56% |
### MM-B (Baseline Multimodal) Results
**Train Performance**

| Avg Loss | Weighted Accuracy | Unweighted Accuracy | Anger Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|---------|------------------|---------------------|----------------|------------------------|------------------|--------------|
| 0.6243  | 77.79%           | 78.24%              | 83.14%         | 72.92%                | 78.53%           | 78.38%       |

**Test Performance**

| Avg Loss | Weighted Accuracy | Unweighted Accuracy | Anger Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|---------|------------------|---------------------|----------------|------------------------|------------------|--------------|
| 0.6544  | 75.72%           | 75.99%              | 77.33%         | 75.14%                | 74.47%           | 77.00%       |


### MMMLA (multimodal multi level attention) Model Results

**Train Performance**

| Avg Loss | Weighted  Accuracy | Unweighted  Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|----------------|-------------------------|----------------------------|------------------------|------------------------|------------------------|------------------------|
|  0.5157         | 81.45%                  | 81.94%                     | 84.62%                 | 80.86%                 | 78.53%                 | 83.73%                 |

**Test Performance**

|  Avg Loss | Weighted  Accuracy | Unweighted  Accuracy | Angry Acc. | Happy/Excited Acc. | Neutral Acc. | Sad Acc. |
|---------------|------------------------|---------------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| 0.5862        | 77.72%                 | 78.31%                    | 83.11%                | 75.14%                | 75.98%                | 79.00%                |

### 4. Analysis.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

### 5. Demo.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

