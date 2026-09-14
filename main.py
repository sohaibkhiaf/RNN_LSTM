import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
import nltk
from Tools.scripts.make_ctype import values
from nltk import sentiment
from nltk.corpus import stopwords
from collections import Counter
import re
from string import punctuation

from pathlib import Path

print(f"Torch version: {torch.__version__}")
print("\n\n")

RANDOM_SEED= 42

# device configuration ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")

# loading data ===========================
df = pd.read_csv("imdb_train_dataset.csv")

print(f"Data shape: {df.shape}")
print(f"Number of positive samples: {df[df['sentiment'] == 'positive'].shape[0]}")
print(f"Number of negative samples: {df[df['sentiment'] == 'negative'].shape[0]}")
print(f"Data samples: {df.head()}")
print("\n\n")

X, y = df['review'].values, df['sentiment'].values

# plot positive vs negative value counts =====================
dd = pd.Series(y).value_counts()
sns.barplot(x=np.array(['negative','positive']),
            y=dd.values,
            hue= np.array(['negative','positive']),
            palette=["red", "green"])
plt.show()


# preprocess string function ====================
def preprocess_string(s):
    # remove !  ?  .  ,  ;  :  '  "
    s = re.sub(r"[^\w\s]", '', s)
    # remove digits 1 4 5 6 0
    s = re.sub(r"\d", '', s)

    # e.g. s= "Amazing!!123" => s= "Amazing"
    return s

# create vocabulary ==================================
MAX_VOCAB_SIZE = 120

word_list = []

# remove stop words (the, is , a, an, of, to, in, and, ...)
# + preprocess words
stop_words = set(stopwords.words('english'))
for text in X:
    for word in text.lower().split():
        word = preprocess_string(word)
        if word not in stop_words and word != '':
            word_list.append(word)

# calculate each word's count in a python dict
word_count = Counter(word_list)

# sorting on the basis of most common words
vocabulary = sorted(word_count, key=word_count.get, reverse=True)[:MAX_VOCAB_SIZE]

# creating a dict
word_to_id = {w: i + 1 for i, w in enumerate(vocabulary)}

print(f"Vocabulary size: {len(word_to_id)}")
print(f"Vocabulary: \n{word_to_id}")
print("\n\n")

# tokenize ====================================
X_encoded = []

# encode  sequences
for text in X:
    X_encoded.append([word_to_id[preprocess_string(word)] for word in text.lower().split()
                            if preprocess_string(word) in word_to_id.keys()])
# encode labels
y_encoded = [1 if label == 'positive' else 0 for label in y]

print(f"X encoded samples: \n{X_encoded[:5]}")
print(f"y encoded samples: \n{y_encoded[:5]}")
print("\n\n")



# plot review length vs frequency =====================
review_len = [len(review) for review in X_encoded]

plt.figure(figsize=(10, 6))

plt.hist(review_len, bins=30)

plt.xlabel("Review length (number of tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Review Lengths")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

pd.Series(review_len).describe()

print("\n\n")

# max sequence length based on plotted review lengths
MAX_SEQ_LEN = 12


# outlier review stats =====================
review_lengths = Counter([len(x) for x in X_encoded])
print("Zero-length reviews: {}".format(review_lengths[0]))
print("Maximum review length: {}".format(max(review_lengths)))
print("\n\n")

## remove any reviews/labels with zero length from the X_encoded list ============
print('Number of reviews before removing outliers: ', len(X_encoded))

# get indices of any reviews with length 0
non_zero_idx = [i for i, review in enumerate(X_encoded) if len(review) != 0]

# remove 0-length reviews and their labels
X_encoded = [X_encoded[i] for i in non_zero_idx]
y_encoded = np.array([y_encoded[i] for i in non_zero_idx])

print('Number of reviews after removing outliers: ', len(X_encoded))
print("\n\n")



# add padding function ====================
def add_padding(sequences, seq_len):

  # initialize padded sequences with zeros
  padded = np.zeros(
      (len(sequences), seq_len),
      dtype=int
  )

  for i, review in enumerate(sequences):
    # truncate if review is longer than seq_len
    truncated_review = np.array(review)[:seq_len]

    if len(review) <= 0:
        continue

    # left padding
    padded[i, -len(review):] = truncated_review

  return padded



# add left padding =====================================
# we have very less number of reviews with length > MAX_SEQ_LEN
# so we will consider only those below it
X_pad = add_padding(X_encoded, MAX_SEQ_LEN)

print(f"Shape of X pad: {X_pad.shape}")
print(f"Shape of y: {y_encoded.shape}")

print(f"X pad:  \n{X_pad}")
print(f"y encoded:  \n{y_encoded}")

print("\n\n")


# train validation split =========================
split_frac = 0.8

## split data into training and validation data (X and y)

split_idx = int(len(X_pad)*split_frac)
X_train, X_val = X_pad[:split_idx], X_pad[split_idx:]
y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]

print(f"X train shape: {X_train.shape}")
print(f"y train shape: {y_train.shape}")
print(f"X val shape: {X_val.shape}")
print(f"y val shape: {y_val.shape}")
print("\n\n")



# create datasets and data loaders =============================
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

BATCH_SIZE = 50

# make sure the SHUFFLE your training data
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)



# obtain one batch of training data =================
X_batch, y_batch = next(iter(train_dataloader))

print('Sample input size: ', X_batch.size()) # batch_size, seq_length
print('Sample input: \n', X_batch)
print()
print('Sample label size: ', y_batch.size()) # batch_size
print('Sample label: \n', y_batch)
print("\n\n")



# sentiment lstm model class ==================================
class SentimentLSTM(nn.Module):
  def __init__(
      self,
      vocab_size = 120+1,
      output_size= 1,
      embedding_dim= 128,
      hidden_dim= 128,
      num_layers= 2,
      dropout_prob=0.3 ):

    super().__init__()

    self.output_size = output_size
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim

    # embedding layer
    self.embedding = nn.Embedding (
        # 121 (120 word+ 1 padding )
        num_embeddings= vocab_size,
        # idx => embedding result size, e.g. 96 becomes [0.34, -0.26, ..., 0.13] (128 values)
        embedding_dim= embedding_dim,
        # padding representation is 0
        padding_idx=0
    )

    #lstm layers
    self.lstm = nn.LSTM(
        input_size=embedding_dim,
        hidden_size= hidden_dim, # hidden state size
        num_layers= num_layers,
        batch_first=True,
        dropout=dropout_prob if num_layers >1 else 0,
    )

    # dropout
    self.dropout = nn.Dropout(p=0.3)

    # fully connected
    self.fc = nn.Linear(in_features= hidden_dim,
                        out_features= output_size )

  def forward(self, x, hidden, cell):

    # embedding layer
    embedded = self.embedding(x)

    # lstm layer
    lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

    # getting the last time step output
    lstm_output = lstm_output[:, -1, :]

    # dropout and fully-connected layer
    output = self.dropout(lstm_output)
    output = self.fc(output)

    return output.squeeze(1), (hidden, cell)

  def init_hidden(self, batch_size):
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

    return (hidden, cell)



# function to calculate accuracy ====================
def accuracy_fn(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


# model evaluation function ========================
def eval_model(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               device = device):

  # move model to device
  model.to(device)

  eval_loss = []
  eval_acc = 0

  # eval mode
  model.eval()

  with torch.inference_mode():
    for X_batch, y_batch in dataloader:

      # init hidden and cell state
      hidden, cell = model.init_hidden(batch_size=X_batch.size(0))

      # move to target device
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      # forward pass
      y_logit , (hidden, cell)= model(X_batch, hidden, cell)

      # calculate loss
      loss = loss_fn(y_logit, y_batch.type(torch.float))
      eval_loss.append(loss.item())

      # calculate accuracy
      acc = accuracy_fn(torch.sigmoid(y_logit), y_batch)
      eval_acc += acc

  loss_value = np.mean(eval_loss)
  acc_value = eval_acc/len(dataloader.dataset)

  print(f'Loss: {loss_value:.4f}')
  print(f'Accuracy: {acc_value*100:.2f}%')


# train step function ==========================
def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               optimizer: torch.optim.Optimizer,
               device = device,
               clip: int = None):
  # train step
  train_loss = []
  train_acc = 0

  # train mode
  model.train()

  for X_batch , y_batch in train_dataloader:
    # init hidden and cell state
    hidden, cell = model.init_hidden(batch_size=X_batch.size(0))

    # move to target device
    X_batch, y_batch = X_batch.to(device),  y_batch.to(device)

    y_logit, (hidden, cell) = model(X_batch, hidden, cell)

    # calculate the loss
    loss = loss_fn(y_logit, y_batch.type(torch.float))
    train_loss.append(loss.item())

    # calculate the accuracy
    acc = accuracy_fn(torch.sigmoid(y_logit), y_batch)
    train_acc += acc
    print(f"Train step: accuracy= {acc/ len(y_batch)*100:.2f}% | loss= {loss.item():.4f}")

    # zero grad
    optimizer.zero_grad()

    # perform backprop
    loss.backward()

    # prevent the exploding gradient problem in rnn/ lstm models
    if clip:
      nn.utils.clip_grad_norm_(model.parameters(), clip)

    # optimizer step
    optimizer.step()

  return  np.mean(train_loss), train_acc/len(train_dataloader.dataset)


# val step function ============================
def val_step(model: nn.Module,
               val_dataloader: torch.utils.data.DataLoader,
               loss_fn : nn.BCEWithLogitsLoss,
               device = device):
  # val step
  val_loss = []
  val_acc = 0

  # eval mode
  model.eval()

  with torch.inference_mode():
    for X_batch, y_batch in val_dataloader:
      # init hidden and cell state
      hidden, cell = model.init_hidden(batch_size=X_batch.size(0))

      # move to target device
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      # forward pass
      y_logit, (hidden, cell) = model(X_batch, hidden, cell )

      # calculate loss
      loss_val = loss_fn(y_logit, y_batch.type(torch.float))
      val_loss.append(loss_val.item())

      # calculate accuracy
      acc_val = accuracy_fn(torch.sigmoid(y_logit), y_batch)
      val_acc += acc_val
      print(f"Validation step: accuracy= {acc_val/ len(y_batch)*100:.2f}% | loss= {loss_val.item():.4f}")

  return np.mean(val_loss), val_acc/len(val_dataloader.dataset)


# train loop function ===========================
def train_model(model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                loss_fn : nn.BCEWithLogitsLoss,
                optimizer: torch.optim.Optimizer,
                device= device,
                epochs: int = 5,
                clip: int = 5):

  train_loss_values, val_loss_values = [],[]
  train_acc_values, val_acc_values = [],[]

  for epoch in range(epochs):

    # train step
    epoch_train_loss, epoch_train_acc = train_step(model= model,
                                      train_dataloader =train_dataloader,
                                      loss_fn= loss_fn,
                                      optimizer= optimizer,
                                      device= device,
                                      clip = clip)
    # val step
    epoch_val_loss, epoch_val_acc = val_step(model=model,
                                      val_dataloader= val_dataloader,
                                      loss_fn= loss_fn,
                                      device= device)

    # append values for plotting
    train_loss_values.append(epoch_train_loss)
    train_acc_values.append(epoch_train_acc)

    val_loss_values.append(epoch_val_loss)
    val_acc_values.append(epoch_val_acc)

    # print epoch loss and acc
    print(f'Epoch: {epoch+1}')
    print(f'Train loss= {epoch_train_loss:.4f} | Validation loss= {epoch_val_loss:.4f}')
    print(f'Train accuracy= {epoch_train_acc*100:.2f}% | Validation accuracy= {epoch_val_acc*100:.2f}%')
    print('='*70)

  return train_loss_values, val_loss_values, train_acc_values, val_acc_values


# plot loss and accuracy function ======================
def plot_results(train_loss_values,
                      val_loss_values,
                      train_acc_values,
                      val_acc_values,
                      epochs):
  plt.figure(figsize = (20, 6))
  # loss
  plt.subplot(1, 2, 2)
  plt.plot(train_loss_values, label='Train loss')
  plt.plot(val_loss_values, label='Validation loss')
  plt.title("Loss")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()
  # accuracy
  plt.subplot(1, 2, 1)
  plt.plot(train_acc_values, label='Train Accuracy')
  plt.plot(val_acc_values, label='Validation Accuracy')
  plt.title("Accuracy")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()

  plt.show()

# train model =============================
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

model = SentimentLSTM(vocab_size=len(vocabulary) +1, # +1 for padding =0
                      output_size=1,
                      embedding_dim=128,
                      hidden_dim=128 ,
                      num_layers=2,
                      dropout_prob=0.3)

#moving to gpu
model.to(device)

print(f"Model: \n{model}")

# loss and optimization functions
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)

print(f"Loss function: \n{loss_fn}")
print(f"Optimizer: \n{optimizer}")

print(f"Evaluation before training: ")
eval_model(model, val_dataloader, loss_fn, device)

# train
print(f"Training loop: ")
train_loss_values, val_loss_values, train_acc_values, val_acc_values = train_model(model,
                                                  train_dataloader,
                                                  val_dataloader,
                                                  loss_fn,
                                                  optimizer,
                                                  device,
                                                  epochs= 5,
                                                  clip= 5
)
print("\n\n")


# plot loss and acc ===========================
plot_results(train_loss_values,
                  val_loss_values,
                  train_acc_values,
                  val_acc_values,
                  epochs= 5)

print("Evaluation after training: ")
eval_model(model, val_dataloader, loss_fn, device)


# predict sentiment function ================================
def predict_sentiment(text: str, model: torch.nn.Module):
  # eval mode
  model.eval()
  with torch.inference_mode():

    # preprocess sequence
    word_seq = np.array([word_to_id[preprocess_string(word)] for word in text.split()
                    if preprocess_string(word) in word_to_id.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)

    # add left padding
    padded = add_padding(word_seq, MAX_SEQ_LEN)

    # convert to tensor
    padded =  torch.from_numpy(padded)

    # move to device
    padded = padded.to(device)

    hidden, cell = model.init_hidden(batch_size=1)

    # calculate probability
    y_logit, (hidden, cell) = model(padded, hidden, cell)
    y_pred= torch.sigmoid(y_logit)
  return y_pred.cpu().detach().numpy()


# test on test dataset ============================
df_test = pd.read_csv("imdb_test_dataset.csv")
X, y = df_test['review'].values, df_test['sentiment'].values

correct = 0
total = len(y)
for i in range(len(y)):

    review = X[i]
    sentiment = y[i]
    sentiment_code = 1 if sentiment == "positive" else 0

    y_prob = predict_sentiment(review, model)
    y_pred = torch.round(torch.from_numpy(y_prob))

    if y_pred == sentiment_code:
        correct += 1
        print(f"Correct prediction => {correct} / {total}")
    else:
        print(f"Incorrect prediction")

print(f"Accuracy: {correct/total*100:.2f}%")
print("\n\n")


# test model with custom text ==================================
while True:
  print("Write a review: ")
  text = input()
  if text == 'exit':
    break

  # make prediction
  y_prob = predict_sentiment(text, model)

  # translate to status
  status = "positive" if y_prob > 0.5 else "negative"

  # percentage negative /positive
  prob = (1 - y_prob.item()) if status == "negative" else y_prob.item()

  print(f'Predicted sentiment is {status} with a probability of {prob*100:.2f}%')

print("\n\n")




# saving model =============================================
# create model dictory path
MODEL_PATH= Path("checkpoints")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# create model save
MODEL_NAME= "model_state_dict.pt"
MODEL_SAVE_PATH= MODEL_PATH/ MODEL_NAME

# save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
print("\n\n")


# loading and evaluating saved model =========================================
torch.manual_seed(42)

loaded_model = SentimentLSTM(vocab_size=len(vocabulary) +1, # +1 for padding =0
                      output_size=1,
                      embedding_dim=128,
                      hidden_dim=128 ,
                      num_layers=2,
                      dropout_prob=0.3)

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model.to(device)

torch.manual_seed(42)

print("Loaded model eval:")
eval_model(
    model=loaded_model,
    dataloader=val_dataloader,
    loss_fn=loss_fn,
    device=device
)
print("\n\n")


