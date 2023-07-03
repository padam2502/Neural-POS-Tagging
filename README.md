
# Neural-POS-Tagging

Neural part-of-speech (POS) tagging is a natural language processing (NLP) task that involves assigning the appropriate part-of-speech tags (such as noun, verb, adjective, etc.) to each word in a given sentence. Neural networks have been widely employed to tackle this task, leveraging their ability to learn intricate patterns and capture contextual information.


## Description

Implemented the following :
- POS Tagger model using LSTM

Designed, implemented and trained a neural sequence model (LSTM) to (tokenize and) tag a given sentence with the correct
part-of-speech tags. For example, given the input

Mary had a little lamb

model should output

Mary  NOUN

had  VERB

a  DET

little  ADJ

lamb  NOUN

Tuned for optimal hyperparameters (embedding size, hidden size, number of layers,
learning rate, complexity of decoding network) and prepared a detailed report on accuracy, precision,
recall and F1-score of the trained model (refer to this function).
Analysed the results (both the scores as well as the optimal hyperparameters).
## Getting Started

### Dependencies

* Python
* Tensorflow/Keras
* nltk
* punkt
* sklearn
* conll_df
* gensim



### Requirements

* Same as mentioned in the assignment
* Need to have metadata pickle files -> word2index, tag2index  and model weights file in the same folder with same name as used in the .py file

Model Link : https://drive.google.com/file/d/1dbwFPtdc3lWORj9Po8DwAk97chAhxhSS/view?usp=sharing



>>>>>>> 2bf77f6 (Add all my files)
