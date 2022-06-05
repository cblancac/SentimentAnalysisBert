![Failed to load image](https://user-images.githubusercontent.com/105242658/168151092-221a7d19-449e-40f5-8800-b68c2aad54a9.jpeg)

# SentimentAnalysisBert

This project trains some models in order to predict the feeling of a given sentence. The sentence is transformed (using BERT) to a tensor, and this tensor can be used by any of the model trained to predict the feeling (POSITIVE or NEGATIVE). The sentence can be written in any language!

![feelings](https://user-images.githubusercontent.com/105242658/169574722-5248f37e-fe35-4564-ad39-4cc78511fccd.png)


## :gear: Setup
- Clone the repository: `https://github.com/cblancac/SentimentAnalysisBert`.
- `pip install pip==20.2`
- `pip install -r requirements.txt`
- Change the line 151 in google_trans_new/google_trans_new.py: 
  `response = (decoded_line + ']')` -> `response = decoded_line`.
- Download the English language model (large ~382 MB): `python -m spacy download en_core_web_lg`.
- (1) Creating the transformation of the texts or (2) training the models is expensive computacionally, so a solution could be use an instance EC2 (Amazon Elastic Compute Cloud). The instance `c5a.8xlarge` can be enough to deal with (1) the full transformation of the raw texts, but only could (2) train the models using a subset of them (if you want to train a model with the complete dataset you would need to move to a bigger instace EC2). Be very careful using this type of payment instances, check the prices initially and turn them off when you are not using them.
- In order to get the tensors which represents the tweets of the train and test dataset, there are two options (due to GitHub doesn't allow to upload such big files):
    - (a) Download the files using the links included in the `download_tensors.txt`. Create a new folder called tensors, unzipped the two files (with `gunzip filename`) and add them in the folder tensors.
    - (b) Generate your own tensors using the script `embedder_bert_texts.py`.
    
    
## :brain: BERT (Bidirectional Encoder Representations from Transformer)

Like other neural networks, Transformer models can’t process raw text directly, so the first step BERT need to do is to convert the text input into numbers that the model can make sense of. It is known as tokenization, which will be responsible for:
- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
- Mapping each token to an integer IDs
- Adding additional inputs that may be useful to the model

BERT receive this list of IDs per sentence and convert them to tensors. It generally has three dimensions:

- **Number of tweets:** The number of tweets given. In this case ~150k for the train dataset and ~62k for the test dataset.
- **Sequence length:** The length of the numerical representation of the sequence (It has been fixed to 32=99.95th percentile over the lengths of all training tweets). The sentences bigger than 32 have been truncated and smaller ones have had padding added, obtaining a length fixed per sentence. 
- **Hidden size:** The vector dimension of each model input. This size may vary from 768 (is common for smaller models) until 3072 or more (in larger models).

All of this can be gotten using the script `embedder_bert_texts.py`. The input recieved, with the tweets, is stored in the folder `data`, which contains a balanced representation between tweets with positive and negative feelings. The output is saved in the folder `tensors`. The shapes of the tensors generated will be (150.000, 32, 768) for the train and (62.000, 32, 768) for the test. 


## 	:weight_lifting_man: Training models

Once the text are converted to tensors, it can feed different neural networks (NN). The output of these NN will be a value between 0 and 1 (the more close to 1 more positive will be the feeling of the sentence). The neurals networks considered here are:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Combination of Gated Recurrent Unit with Convolutional Neural Network (GRU+CNN)

To choose the best combination of some hyperparameter, Random Search strategy has been used. Using the script `train_model.py`, some light models have been trained with 1.000 and 10.000 tweets for every of the NN just mentioned. This models have been saved in the folder `models`. The best combination of the hyperparameters per model are saved in the file `results.csv`. You can train your own models choosing the size and your favourite neural network. 

## :bar_chart: Evaluate the models

In the previous step, the best model per size and NN have been saved. Now it is time to evaluate them over the test dataset. The `evaluate_models.py` is the encharged to do it, and it will save the metrics in the file `model_evaluation.csv`.

The results obtained for this sizes and NN have been:
| Neural Network | Size| Accuracy |
| ----- | ---- | ---- |
| cnn | 1.000 | 72,75 |
| lstm | 1.000 | 72,87 |
| gru | 1.000 | 73,90 |
| gru+cnn | 1.000 | 73,57 |
| cnn | 10.000 | 78,32 |
| lstm | 10.000 | 78,53 |
| gru | 10.000 | 78,47 |
| gru+cnn | 10.000 | 78,58 |

## :tada: Make predictions

Finally the prediction of the feeling of a sentence can be done using the script `predict.py`. Executing this script, we will have to introduce our sentence as input (separated by | if we need to introduce more than one), and it will give us back a dictionary with the next information for every sentence:
- to_en: The tranlation to English of our sentence.
- label: The classification the model made for the sentence, POSITIVE or NEGATIVE
- score: The confidence with which the classification has been made. The closer this value is to one, the more confident the model is of having classified it with the returned label.

Example:
    Running the script, `predict.py`, you will receive a sentence asking you for introduce some sentence. After you type the sentence you will see the output created with the model trained as the dictionary that has been already commented
    

    
```
>>> python predict.py
Introduce here the sentence(s) you want to predict (separate by | if you introduce more than one):
I'm publishing for the first time on GitHub!

[("Today I'm publishing for the first time on GitHub!", {'to_en': "Today I'm publishing for the first time on GitHub!", 'label': 'POSITIVE', 'score': 0.8692686})]
```

    

    


