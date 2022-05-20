![Failed to load image](https://user-images.githubusercontent.com/105242658/168151092-221a7d19-449e-40f5-8800-b68c2aad54a9.jpeg)

# SentimentAnalysisBert

This project trains some models in order to predict the feeling of a given sentence. The sentence is transformed (using BERT) to a tensor, and this tensor can be used by any of the model trained to predict the feeling (POSITIVE or NEGATIVE). The sentence can be written in any language!

![feelings](https://user-images.githubusercontent.com/105242658/169574722-5248f37e-fe35-4564-ad39-4cc78511fccd.png)


## Setup
- Clone the repository: `https://github.com/cblancac/SentimentAnalysisBert`.
- `pip install -r requirements.txt`.
- Change the line 151 in google_trans_new/google_trans_new.py: `response = (decoded_line + ']')` -> `response = decoded_line`.
- Downloaded the english language model (large ~382 MB): `python -m spacy download en_core_web_lg`.
- Create the transformation of the texts or train the models is expensive computacionally, so a solution could be use a instance EC2 (Amazon Elastic Compute Cloud). The instance `c5a.8xlarge` can be enough to deal with the full transformation of the raw texts, but only could train the models using a subset of them (if you want to train a model with the complete dataset you would need to move to a bigger instace EC2).
- In order to get the tensors which represents the tweets of the train and test dataset, there are two options (due to GitHub doesn't allow to upload such big files):
    - (1) Download them using the links included in the `download_tensors.txt`. This two files have to be unzipped (using `gunzip filename`) and included in the folder tensors.
    - (2) Generate your own tensors using the script `embedder_bert_texts.py`.
    
    
##  BERT (Bidirectional Encoder Representations from Transformer)

Like other neural networks, Transformer models can’t process raw text directly, so the first step BERT do is to convert the text inputs into numbers that the model can make sense of. It is known as tokenization, which will be responsible for:
- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
- Mapping each token to an integer IDs
- Adding additional inputs that may be useful to the model

BERT receive this list of IDs per sentence and convert them to tensors. It generally has three dimensions:

- Number of tweets: The number of tweets given. In this case ~150k for the train dataset and ~62k for the test dataset.
- Sequence length: The length of the numerical representation of the sequence (It has be fixed to 32=99.95th percentile over the lengths of all training tweets). The sentence bigger than 32 have been truncated and the smaller have added a padding, obtaining a length fixed per sentence. 
- Hidden size: The vector dimension of each model input. This size may vary from 768 (is common for smaller models) until 3072 or more (in larger models).

All this things can be gotten using the script `embedder_bert_texts.py`. The input recieved, with the tweets, is stored in the folder `data`, which contains a balanced representation between tweets with positive and negative feelings. The output is saved in the folder `tensors`. The shapes of the tensors generated will be train: (150.000, 32, 768) and test: (62.000, 32, 768). 






