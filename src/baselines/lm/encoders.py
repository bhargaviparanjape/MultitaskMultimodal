
from model import *

class LookupEncoder(nn.Module):
    def __init__(self, vocab_size, word_dim,pre_trained_embeddings=None):
        super(LookupEncoder,self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_dim)
        if pre_trained_embeddings:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))

    def forward(self, input):
        return self.embedding_layer(input)


class Encoder(nn.Module):
    def __init__(self, word_vocab_size, input_size, hidden_size,num_layers,rnn_type, embeddings = None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.word_vocab_size = word_vocab_size

        ## Define Layers
        self.embedding_layer = LookupEncoder(word_vocab_size, input_size, embeddings)
        if rnn_type == 'LSTM':
            self.network = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

        elif rnn_type == 'GRU':
            self.network = nn.GRU(input_size, hidden_size, num_layers=num_layers)


    def forward(self, input_sequence, lengths, hidden):
        embeds = self.embedding_layer.forward(input_sequence)
        output, hidden = self.network(embeds, hidden)  # output: concatenated hidden dimension
        return output, hidden





