## Business Understanding

When working as a customer service agent, you are bound to run into the whole spectrum of emotions - annoyance, gratefulness, anger, or complete neutrality. These agents are often monitored, scored on how well - and how many issues they address. 

A satisfied customer - either leaving a positive review, or having the call recorded and review - will leave a good impression on the agent. 
A frustrated customer is more likely to leave a negative review, or a manager reviews a call that went poorly - it reflects negatively on the agent. 

Implementing AI into the customer support pipeline can be beneficial in many ways:

- If an agent knows as soon as their customer is upset, they can adjust how they speak to them, or you can even adapt the script the agent uses as the customer’s emotions fluctuate. 
- The agent feels more secure in their responses, or know that they can escalate to someone better equipped to handle difficult customers. 
- The customer feels heard, finds a resolution faster, and has an overall better experience!

#### Objective: Create a model to identify customer emotion (Upset/Not Upset) over the phone.

---

## Data Understanding

In this project, we are dealing with four datasets containing English audio recordings in the .wav format. Each audio recording is labeled with an emotion that the speaker is evoking in their statement. Our goal is to build a model that can successfully map an emotion to a given voice clip of someone speaking.

> Compiled datsets can be found on [Kaggle: Speech / Emotion Recognition](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en).

To achieve this, we will extract various features from the audio recordings that are relevant for analyzing speech and emotion. Here are the features we will be working with and their significance for working with audio:

1. **Mel-frequency Cepstral Coefficients (MFCCs):** MFCCs are a compact representation of the short-term power spectrum of a sound. They are widely used in speech recognition and audio analysis tasks because they capture the essential characteristics of the audio signal while being robust to noise and other variabilities. MFCCs are particularly useful for identifying the phonetic content of speech, which can be helpful in determining the emotional state of the speaker.

2. **Spectral Centroid:** The spectral centroid is a measure of the brightness or sharpness of a sound. It represents the weighted mean frequency of the spectrum and can be used to distinguish between different types of sounds or emotions. For example, a bright, harsh sound might have a higher spectral centroid than a mellow, soft sound.

3. **Chroma Features:** Chroma features describe the distribution of energy across different pitch classes (notes) in the audio signal. They are useful for capturing tonal information, which can be relevant for identifying emotions in speech, particularly those related to intonation patterns and stress.

4. **Zero-Crossing Rate:** The zero-crossing rate is a measure of the number of times the audio signal crosses the zero amplitude axis within a given time frame. It can be used to distinguish between different types of sounds, such as voiced and unvoiced speech, and can provide insights into the energy distribution of the audio signal.
RMS Energy: The Root Mean Square (RMS) energy is a measure of the overall energy or loudness of an audio signal. It can be useful for detecting variations in volume or intensity, which can be indicative of certain emotions, such as anger or excitement.

> By extracting and analyzing these features, we can capture various acoustic characteristics of the speech signal that may be relevant for distinguishing between different emotions. This multi-faceted approach can provide a more comprehensive representation of the audio data, potentially leading to better performance in the emotion classification task.

---

#### Features:

| **Feature**            | **Description**                                                                                      |
|------------------------|-----------------------------------------------------------|
| **id**                 | Unique identifier for each audio sample.                                                             |
| **filename**           | Name of the audio file.                                                                              |
| **emotion**            | The labeled emotion expressed in the audio (e.g., angry, happy).                                     |
| **path**               | File path to the location of the audio file.                                                         |
| **mfccs**              | A list of 13 Mel-frequency Cepstral Coefficients (MFCCs), representing the audio's short-term power spectrum. Useful for identifying speech characteristics. |
| **chroma**             | A 12-dimensional representation of the energy distribution across different pitch classes (notes), capturing tonal information. |
| **spectral_centroid**  | Represents the "center of mass" of the audio spectrum, indicating the brightness or sharpness of the sound. |
| **zero_crossing_rate** | A measure of how often the audio signal crosses the zero amplitude axis, indicating the noisiness or texture of the sound. |

### Data Limitations
- **Imbalanced Distribution** - The uneven distribution of positive vs. negative emotions could cause the model to favor the majority class, making it less effective at detecting emotions in underrepresented categories (e.g., fewer positive or neutral emotions).
- **Limited Dataset Size** - A small dataset restricts the model’s ability to generalize to new, unseen data, potentially leading to overfitting and poor performance on real-world data. This is especially critical if the dataset doesn't cover a wide range of speakers or emotional expressions.
- **Speaker Variability** - Emotional expression can vary greatly between different individuals, and if the dataset doesn't capture enough speaker diversity (e.g., accents, vocal tones), the model may struggle to accurately detect emotions across a broader population.
- **Emotion Subjectivity** - Emotions are inherently subjective, and the labeled emotions in the dataset may not always perfectly match how different people perceive or express emotions. This could introduce noise into the labels, affecting the model’s ability to learn accurately.

---

## Data Preperation

First, we will combine our 4 datasets in English: Crema, Ravdess, Savee and Tess. Each of them contains audio in .wav format with some main labels. (Note - not all datasets represent the same emotions, we will clean up the data labels to be as generic / inclusive as possible.

We will pull each dataset into their own dataframe, making note of *where* the file is, so we can later pull our features from each audio file. 

Then, we will merge them all into one dataframe and extract our audio features as mentioned earlier:

- Mel-frequency cepstral coefficients (MFCCs)
- Spectral centroid
- Chroma features
- Zero-crossing rate

---

## Modeling

Multiple models were created and built upon iteratively. A few models attempted to use augmented audio to even out the distribution (synthetic data). In the end, we were able to create a multiclass classification model with 60% overall accuracy, and a binary classification model with 80% accuracy. Below are their classification reports:

### Multiclass Classification:
> 6 classes: angry, happy, sad, neutral, disgust, fear

Classification Report:

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **angry**      | 0.69      | 0.72   | 0.71     | 362     |
| **disgust**    | 0.61      | 0.48   | 0.54     | 385     |
| **fear**       | 0.68      | 0.41   | 0.51     | 381     |
| **happy**      | 0.55      | 0.67   | 0.61     | 448     |
| **neutral**    | 0.58      | 0.62   | 0.60     | 340     |
| **sad**        | 0.57      | 0.71   | 0.63     | 408     |
| **surprise**   | 0.61      | 0.46   | 0.52     | 37      |
| **Accuracy**   |           |        | 0.60     | 2361    |
| **Macro Avg**  | 0.61      | 0.58   | 0.59     | 2361    |
| **Weighted Avg** | 0.61    | 0.60   | 0.60     | 2361    |

This model was successfully able to differentiate these emotions with 60% accuracy (much better than random guessing - which would be ~17%). 

Though, this score isn't incredibly impressive, and the difference between a customer being angry or disgusted wouldn't make a huge difference to the CS agent. Instead, we will create a binary-classification problem.

### Binary Classification:

> 0: **UPSET** and 1: **NOT UPSET**

With the f1-score of 86% for detecting negative emotions, and 68% for detecting positive / neutral emotions: our model is definitely capable of flagging customers when they are upset or not on the phone! This discrepency is likely due to the imbalance of positive/negative values in our dataset. I chose not to augment positive samples for this model, as it seemed to add noise and confusion.

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Negative** | 0.80      | 0.94   | 0.86     | 1585    |
| **Positive** | 0.81      | 0.51   | 0.63     | 776     |
| **Accuracy** |           |        | 0.80     | 2361    |
| **Macro Avg** | 0.80      | 0.73   | 0.75     | 2361    |
| **Weighted Avg** | 0.80      | 0.80   | 0.79     | 2361    |

---

## Evaluation of Final Model

#### Binary Classification


---
## Evaluation of Final Model: Binary Classification

#### Justification

For this emotion detection model, **precision**, **recall**, and **f1-score** are crucial metrics, as both false-positives (incorrectly flagging calm customers) and false-negatives (missing upset customers) can impact customer satisfaction.

- **Precision**:
  - **Negative (calm)**: 80% of predicted calm customers were actually calm.
  - **Positive (upset)**: 81% of predicted upset customers were actually upset.
  
- **Recall**:
  - **Negative (calm)**: 94% of calm customers were correctly identified.
  - **Positive (upset)**: 51% of upset customers were correctly identified.
  
- **F1-Score**:
  - **Negative (calm)**: 86%
  - **Positive (upset)**: 63%

This model is effective for:
- **Operational Efficiency**: The model can alert agents when a customer is upset, allowing them to adjust their approach in real-time, improving customer experience.
- **Escalation Control**: By tracking how long a customer remains upset, the model can help escalate calls to specialized teams, ensuring that frustrated customers receive the attention they need before issues escalate further.
- **Agent Training and Feedback**: The model provides valuable performance metrics, which can be used to give specific feedback to agents on how they handle both positive and negative calls, improving overall service quality.

#### Recommendations:

- **Alert the agent**: Implement real-time emotion detection to notify agents when a customer is upset, allowing them to adjust their approach and improve the interaction.
  
- **Track escalation**: Monitor how long a customer remains upset and, if necessary, escalate the call to a higher level of support. Consider creating a specialized team trained to handle prolonged negative customer interactions.

- **Provide performance feedback**: Use this model to track both customer emotions and agent performance. Leverage these insights to provide specific feedback to agents, helping them improve their handling of positive versus negative calls.

- **Enhance agent training**: Incorporate this tool into your training process for new employees, offering direct guidance on managing different emotional states during calls.

---

## Repository

#### To replicate:
- download the datset from [Kaggle: Speech / Emotion Recognition](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) - save as 'dataset.zip' - the notebook will handle extraction!

> GitHub file size limits prevent me from uploading the dataset directly.

- [Jupyter Notebook](notebook.ipynb)
- Presentation [slides](slides.pdf)