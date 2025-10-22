# GE2PE: Context-based Persian Grapheme-to-Phoneme Conversion Using Sequence-to-Sequence Models  
Many Text-to-Speech (TTS) systems, particularly in low-resource environments, struggle to produce natural and intelligible speech from grapheme sequences. One solution to this problem is to use Grapheme-to-Phoneme (G2P) conversion to increase the information in the input sequence and improve the TTS output. However, current G2P systems are not accurate or efficient enough for Persian texts due to the language’s complexity and the lack of short vowels in Persian grapheme sequences. In our study, we aimed to improve resources for the Persian language. To achieve this, we introduced two new G2P training datasets, one manually-labeled and the other machine-generated, containing over five million sentences and their corresponding phoneme sequences. Additionally, we proposed two new evaluation datasets for Persian sub-tasks such as Kasre-Ezafe detection, homograph disambiguation, and out-of-vocabulary words. Finally, we developed a new sentence-level end-to-end model to address the challenges of the Persian language. This model was trained using a two-step method, introduced in this thesis, to maximize the impact of manually-labeled data. Our results showed that our model outperformed the state-of-the-art by 0.04% in PER, 1.86% in WER, 4.03% in Kasre-Ezafe Recall, and 3.42% in homograph disambiguation accuracy using the data and metrics proposed in this work.  
**Keywords**: Grapheme-to-Phoneme Conversion, End-to-End Model, Semi-supervised Learning, Transformer  

## Data
**Training Data**  
We use two datasets for training:  
1. FarsDat_aligned, available in the data directory of this project.  
2. Machine_generated, available using the link provided in the data directory.
  
**Evaluation Data**  
We use two datasets for evaluation:  
1. Kasre_test, available in the data directory of this project.  
2. Homograph_test, also available in the data directory of this project.

## Training  
For pre-training and fine-tuning the model you can use the training notebook.

## Testing  
The GE2PE.py file contains the final module introduced in this thesis. You can use the G2P_base notebook to download all the requirements and generate output using our final module.  

## Models  
The final pre-trained and fine-tuned versions of GE2PE are accessible trough the links provided in Models.txt file.

