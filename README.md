# Chatbot_using_RNN:

1. **Installing Dependencies:**
```python
!pip install opendatasets --quiet
```
This line uses the `!pip` command to install the `opendatasets` package, which is used to download datasets.

2. **Importing Libraries:**
```python
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```
These lines import the necessary libraries and modules for data processing, modeling, and training. Some notable libraries used are `nltk` for natural language processing, `numpy` for numerical operations, and `torch` for creating and training neural networks.

3. **Downloading NLTK Data:**
```python
nltk.download('punkt')
```
This line downloads the necessary data for tokenization using the `nltk` library. The `'punkt'` data is used for tokenizing sentences into words.

4. **Tokenization and Stemming Functions:**
```python
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())
```
These functions are used for tokenizing sentences into words and applying stemming to each word. Stemming reduces words to their root form, so different variations of a word are treated as the same word (e.g., "running" and "run" both become "run").

5. **Bag-of-Words Function:**
```python
def bagOfWord(tokenized_sentence, words):
    sentenceWords = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentenceWords:
            bag[idx] = 1
    return bag
```
This function creates a bag-of-words representation for a given sentence. It takes tokenized words and a list of unique words as input. It returns a binary vector (numpy array) where each element represents whether a word from the vocabulary exists in the sentence or not.

6. **Loading and Preprocessing the Dataset:**
```python
DATASET_URL = '/content/intents.json'

with open(DATASET_URL, 'r') as file:
    intents = json.load(file)

allWords = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))

ignoreWords = ['?', '.', '!']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
allWords = sorted(set(allWords))
tags = sorted(set(tags))
```
This section loads a JSON file containing training data for the chatbot. Each intent in the dataset consists of a tag and a list of patterns representing user queries. The code tokenizes the patterns, performs stemming on the words, and creates a list of all unique words (`allWords`). It also collects pairs of tokenized patterns and their corresponding tags (`xy`).

7. **Creating Training Data:**
```python
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWord(pattern_sentence, allWords)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
```
This section prepares the training data for the neural network. It converts the tokenized patterns into bag-of-words representations using the `bagOfWord` function. The labels are encoded as numerical values using the index of the tag in the sorted `tags` list.

8. **Defining Hyperparameters:**
```python
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
```
These variables define the hyperparameters for training the neural network. `num_epochs` represents the number of training epochs, `batch_size` determines the number of samples per batch, `learning_rate` controls the learning rate of the optimizer, `input_size` represents the size of the input features, `hidden_size` specifies the number of neurons in the hidden layer, and `output_size` is the number of output classes.

9. **Creating a Custom Dataset:**
```python
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
```
This code defines a custom `Dataset` class to load the training data. It implements the required methods (`__init__`, `__getitem__`, and `__len__`) to make the dataset compatible with PyTorch's `DataLoader`. The training data is loaded into the `ChatDataset` object, and a `DataLoader` is created to iterate over the data in batches during training.

10. **Defining the Neural Network Model:**
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
This code defines a neural network model for the chatbot. It is a simple feedforward network with two hidden layers. The `NeuralNet` class inherits from `nn.Module` and implements the `forward` method, which defines the flow of data through the network. The model, loss function (`CrossEntropyLoss`), and optimizer (`Adam`) are instantiated.

11. **Training the Neural Network:**
```python
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final Loss: {loss.item():.4

f}')
```
This code performs the training loop for the neural network. It iterates over the training data in batches, computes the forward pass, calculates the loss, performs backpropagation, and updates the model parameters. The loss value is printed every 100 epochs, and the final loss is displayed at the end of training.

12. **Saving the Trained Model:**
```python
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": allWords,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training Complete. Model Saved To {FILE}')
```
After training, the model's state dictionary, along with other relevant information, is saved to a file named `data.pth` using the `torch.save()` function. This saved file can be loaded later to use the trained model without retraining.

13. **Loading the Trained Model:**
```python
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
model.load_state_dict(model_state)
model.eval()
```
To use the trained model, the code first loads the previously saved information from `data.pth`. It then creates a new instance of the `NeuralNet` model, loads the saved model's state dictionary, and sets the model to evaluation mode (`model.eval()`).

14. **Chatting with the Chatbot:**
```python
bot_name = "Baymax"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bagOfWord(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
```
This section allows interactive chatting with the chatbot. It takes user input, tokenizes it, converts it into a bag-of-words representation, and feeds it to the trained model. The model predicts the appropriate tag for the input, and if the confidence is above a certain threshold (0.75 in this case), a random response from the corresponding intent is printed. Otherwise, the chatbot indicates that it does not understand the input.

This code provides a basic structure for building a chatbot using a simple neural network architecture. It demonstrates how to preprocess data, train a model, save the model's parameters, load a saved model, and use the model for inference. However, note that the code assumes the existence of a JSON dataset file (`intents.json`) containing the training data in a specific format. Without the actual dataset, it is not possible to execute the code as-is.
