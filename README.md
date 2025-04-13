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

The data file `data_processed.pkl` (672.3MB) is too large to be uploaded to GitHub. This file is created after running the `DataProcessing.ipynb` notebook (on the IEMOCAP dataset), which preprocesses the data and extracts features. You can either run the notebook to create the pickle file or download it from Google Drive using the following link:

[Download data_processed.pkl from Google Drive](<https://drive.google.com/file/d/1OCHS_sikOoL0ZFbO39F54vd61k8RWsot/view?usp=sharing>)

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

#### Results:
- The final dataset will be stored in the data DataFrame and is ready for further processing and analysis.

### 2. UnimodalClassifier.ipynb

#### Description:

We focus on implementing and testing different BLSTM-based unimodal classifiers for lexical and acoustic data. It includes the following key components:

- Loading libraries and preprocessing data.
- Implementing multiple BLSTM-based classification models.
- Exploring different pooling techniques for feature extraction and classification.

We implement several BLSTM architectures tailored for unimodal classification tasks. These models differ in how they process sequence outputs:

- Model I: BLSTM Last Block Output
This model uses the output of the final BLSTM block for classification.
    - It packs padded sequences for efficient processing, extracts the output corresponding to the last timestep of each sequence, and passes it through a fully connected layer for classification.
- Model II: Averaging Pooling
    - Here, average pooling is applied across all timesteps of the BLSTM outputs. The averaged features are used as input to a linear layer for classification.
- Model III: Context-Based Attention Pooling
    - This model incorporates attention mechanisms to compute weighted averages of BLSTM outputs based on learned context weights.

A custom attention module (ContextBasedAttention) calculates attention scores, which are used to extract contextually relevant features for classification.

#### Results:

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

### 3. MMClassifier.ipynb

#### Description:

- Acoustic Models

    Three BLSTM-based models are implemented for acoustic feature classification:
    - Last Block Output: Uses the output of the final BLSTM block for classification.
    - Average Pooling: Applies average pooling across all BLSTM outputs.
    - Attention-Based: Uses context-based attention pooling to focus on important parts of the sequence.

- Lexical Models

Similar BLSTM-based models are implemented for lexical feature classification, with input embeddings extracted from a pre-trained BERT model.

- Multimodal Models
    - Baseline Multimodal Model (B-MM):
        - Combines acoustic and lexical BLSTM outputs using concatenation.
        - A fully connected layer is used for final classification.
    - Gated Multimodal Unit (GMU):
        - Implements modality-based attention using GMU to prioritize one modality over the other.
        - Includes low-level attention mechanisms for both modalities.
        - Outputs are fused using GMU before classification.

Models are trained using CrossEntropyLoss and optimized with Adam optimizer.




#### Results:

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

#### Description:

- Attention Values Plot

Plots attention values for acoustic (A) and lexical (L) modalities.

    def plot_attention_values(att_acoustic, att_lexical):
        plt.bar(['A', 'L'], [att_acoustic, att_lexical], color=['blue', 'orange'])
        plt.title('Attention Values')
        plt.ylabel('Attention')
        plt.show()



- Plot Prediction Probabilities

Visualizes probabilities for classes: Angry (A), Happy/Excited (H/E), Neutral (N), Sad (S).
    
    def plot_prediction_probabilities(probabilities):
        plt.bar(['A', 'H/E', 'N', 'S'], probabilities, color=['red', 'orange', 'grey', 'cyan'])
        plt.title('Prediction Probabilities')
        plt.ylabel('Probability')
        plt.show()


- Acoustic vs Lexical Attention:

Acoustic attention contributes ~45%, while lexical attention contributes ~55%.
    
    Acoustic Attention Mean: 0.45218852
    Lexical Attention Mean: 0.54781148 (Calculated as `1 - Acoustic Attention`)




#### Conclusion: 

This analysis provides insights into the model's behavior:

1. Accurate predictions for neutral emotions.
2. Challenges in distinguishing ambiguous samples.
3. High reliance on lexical modality.


The visualizations help identify areas for improvement in speech emotion recognition models!



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
