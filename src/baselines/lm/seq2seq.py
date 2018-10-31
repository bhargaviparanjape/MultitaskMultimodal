
from model import Model
from encoders import *
from decoders import *

class Seq2Seq(Model):
    def __init__(self,args, data_loader):
        super(Seq2Seq,self).__init__(args, data_loader)
        self.criterion = nn.CrossEntropyLoss()
        self.encoder = Encoder(self.word_vocab_size, self.word_dim, self.hidden_dim, self.num_layers, args.rnncell)
        self.decoder = Decoder(self.hidden_dim, self.word_vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        if self.is_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.hidden = self.initHidden()

    def forward(self,input,input_length, target_variable):
        hidden = self.repackage_hidden(self.hidden)  # To avoid backpropagation to the very beginning
        encoder_output, encoder_hidden = self.encoder(input, input_length, hidden)    #(time_steps, batch_size, hidden_dim)  (num_layers, batch_size, hidden_dim)
        self.hidden = encoder_hidden
        encoder_output = self.dropout(encoder_output)

        decoder_output = self.decoder(encoder_output)  #(time_steps, batch_size, vocab)

        target = torch.squeeze(target_variable.view(-1, self.batch_size * self.bptt))

        loss = self.criterion(decoder_output.view(-1, self.word_vocab_size), target)
        return loss

    def repackage_hidden(self, hidden):
        if type(hidden) == autograd.Variable:
            return autograd.Variable(hidden.data)


    def initHidden(self):
        result = autograd.Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_dim))  # (batch_size, hidden_dim)
        return result
