# Text Summarization with PEGASUS

This project aims to fine-tune and utilize the PEGASUS model from Google for text summarization tasks, specifically focusing on summarizing dialogues from the SAMSUM dataset. It leverages PyTorch, Transformers, and several other libraries to preprocess, fine-tune, evaluate, and generate summaries.

## Installation

Clone this repository and install the required packages using pip:

!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr
!pip install transformers accelerate


# Dependencies
Python 3.x
PyTorch (It will utilize GPUs if available)
Transformers
Datasets
NLTK
Pandas
Matplotlib
Py7zr

# Dataset
The model is fine-tuned on the SAMSUM dataset, which consists of conversations and their summaries.
Dataset link : https://huggingface.co/datasets/samsum

# Pre-trained Model
The pre-trained model used is Google's PEGASUS-cnn_dailymail, designed for summarization tasks.
model link : https://huggingface.co/google/pegasus-cnn_dailymail?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct


# Fine-tuning
The model is fine-tuned on a subset of the SAMSUM dataset for demonstration purposes. For more robust training, consider using a larger portion of the dataset.

# Evaluation
The model's performance is evaluated using the ROUGE metric, specifically ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum.

# Usage
After fine-tuning, the model can be used to summarize new dialogues.

# Saving and Loading the Model
Instructions are provided for saving the fine-tuned model and tokenizer, as well as loading them for future use.

