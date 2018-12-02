import torch
import torch.nn as nn


class ContextAttention(nn.Module):

    def __init__(self, context_dim, query_dim, attention_type='general'):
        super(ContextAttention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.context_dim = context_dim
        self.query_dim = query_dim
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(context_dim, query_dim, bias=False)

        self.linear_out = nn.Linear(query_dim * 2, query_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, self.context_dim)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, self.query_dim)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * self.query_dim)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, self.query_dim)
#         output = self.tanh(output)

        return output, attention_weights
