from model import *

class Decoder(nn.Module):
    def __init__(self, hidden_size, word_vocab_size):
        super(Decoder,self).__init__()
        self.output_layer = nn.Linear(hidden_size, word_vocab_size)

    def forward(self, input):
        return self.output_layer(input)