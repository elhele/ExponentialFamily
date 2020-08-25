import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.distributions as tdist
from datasets.torchtext_preprocessing import TextPreprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ExponentionalFamilyModel(nn.Module):
    def __init__(self, sentence_len = 10, window_size = 2, dictionrary_len = 100, embedding_size = 30, distr_type = "bernoulli"):
        super().__init__()
        self.dictionrary_len = dictionrary_len
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.distr_type = distr_type
        self.sentence_len = sentence_len
        self.mu = 0
        self.sigma = 0.1
        self.rho = nn.Parameter(torch.normal(self.mu, self.sigma, size=(embedding_size, dictionrary_len))) #nn.Parameter(tdist.Normal(torch.tensor([mu]), torch.tensor([sigma])).sample((embedding_size, dictionrary_len))) #
        self.alpha = nn.Parameter(torch.normal(self.mu, self.sigma, size=(embedding_size, dictionrary_len)))

    def loss_bernoulli(self, sentence_dictionary_indices):
        loss = 0
        for word_index in range(self.sentence_len):
            context_sum = 0
            for context_word_index in range(word_index-window_size, word_index+window_size):
                if context_word_index != word_index:
                    context_sum += torch.dot(alpha, sentence_dictionary_indices[context_word_index])
            eta = torch.log(torch.dot(torch.transpose(rho, 0, 1), context_sum))
            t = self.sentence_dictionary_indices[word_index]
            etaT = torch.transpose(eta, 0, 1)
            a = torch.log(1 + eta)
            loss += torch.dot(etaT, t) - a
        return loss

    def loss_gaussian(self, sentence_dictionary_indices):
        return None

    def forward(self, sentence_dictionary_indices, lengths=None):
        log_probability = torch.sum(tdist.Normal(loc = self.mu, scale = self.sigma).log_prob(self.rho) + tdist.Normal(loc = self.mu, scale = self.sigma).log_prob(self.alpha))
        if self.distr_type == "bernoulli":
            return self.loss_bernoulli(sentence_dictionary_indices) + log_probability
        elif self.distr_type == "gaussian":
            return self.loss_gaussian(sentence_dictionary_indices) + log_probability

preprocessing = TextPreprocessing()
train_text = preprocessing.build_text()
print(train_text)
vocab, embed =  preprocessing.build_vocabulary()
dictionrary_len = len(vocab.itos)
embedding_dim = 50
model = ExponentionalFamilyModel(sentence_len = 10, dictionrary_len = dictionrary_len).to(device)
print(model.state_dict())

lr = 1e-1
n_epochs = 1000
optimizer = optim.Adagrad(model.parameters(), lr=lr)



# for epoch in range(n_epochs):
#     model.train()
#     loss = model(sentence)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()










#
#
# ### Linear Regression test
#
# # Data Generation
# np.random.seed(42)
# x = np.random.rand(100, 1)
# y = 1 + 2 * x + .1 * np.random.randn(100, 1)
#
# # Shuffles the indices
# idx = np.arange(100)
# np.random.shuffle(idx)
#
# # Uses first 80 random indices for train
# train_idx = idx[:80]
# # Uses the remaining indices for validation
# val_idx = idx[80:]
#
# # Generates train and validation sets
# x_train, y_train = x[train_idx], y[train_idx]
# x_val, y_val = x[val_idx], y[val_idx]
#
# # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# # and then we send them to the chosen device
# x_train_tensor = torch.from_numpy(x_train).float().to(device)
# y_train_tensor = torch.from_numpy(y_train).float().to(device)
#
#
# class ManualLinearRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
#         self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#         self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#
#     def forward(self, x):
#         # Computes the outputs / predictions
#         return self.a + self.b * x
#
#
# torch.manual_seed(42)
#
# # Now we can create a model and send it at once to the device
# model = ManualLinearRegression().to(device)
# # We can also inspect its parameters using its state_dict
# print(model.state_dict())
#
# lr = 1e-1
# n_epochs = 1000
#
# loss_fn = nn.MSELoss(reduction='mean')
# optimizer = optim.SGD(model.parameters(), lr=lr)
#
# for epoch in range(n_epochs):
#     # What is this?!?
#     model.train()
#
#     # No more manual prediction!
#     # yhat = a + b * x_tensor
#     yhat = model(x_train_tensor)
#
#     loss = loss_fn(y_train_tensor, yhat)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
# print(model.state_dict())