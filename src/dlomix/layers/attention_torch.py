import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderAttentionLayer(nn.Module):
    """
    Decoder attention layer.

    This layer computes attention weights over the time steps for each feature and then
    multiplies the input by these weights. It is assumed that the input tensor has shape
    (batch_size, time_steps, features), and that the provided `time_steps` matches the
    size of the time dimension.

    Parameters
    ----------
    time_steps : int
        Number of time steps in the input data (i.e. the sequence dimension)
    """

    def __init__(self, time_steps):
        super(DecoderAttentionLayer, self).__init__()
        self.time_steps = time_steps
        # This linear layer maps a vector of length time_steps to a vector of length time_steps.
        self.linear = nn.LazyLinear(time_steps)

    def forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, features).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input.
        """
        # Permute the tensor from (batch, time_steps, features) to (batch, features, time_steps)
        x_permuted = x.transpose(1, 2)

        # Apply the linear layer on the last dimension (which has size time_steps)
        # and then apply softmax along that dimension.
        attn = self.linear(x_permuted)
        attn = F.softmax(attn, dim=-1)

        # Permute the attention tensor back to (batch, time_steps, features)
        attn = attn.transpose(1, 2)
        # Multiply the original input with the attention weights elementwise

        out = x * attn
        return out


class AttentionLayer(nn.Module):
    """
    Attention layer.

    This layer computes a weighted average over time steps of the input sequence.
    It first computes an “attention score” for each time step by performing a dot
    product between the input at that time step and a learned weight vector (and optionally
    adding a learned bias per time step). It then applies a tanh nonlinearity (unless
    the `context` flag is True, in which case the tanh result is replaced by another dot
    product with a learned context vector), exponentiates the result, applies an optional mask,
    normalizes the scores to form a probability distribution over time, and then returns
    the weighted sum over the time dimension.

    Parameters
    ----------
    feature_dim : int
        The number of features (i.e. the last dimension of the input tensor).
    seq_len : int
        The fixed length of the time dimension of the input.
    context : bool, optional
        Whether to use a separate context vector. If True, the attention score is computed as
        a dot product with this vector (ignoring the tanh of the weighted sum). Defaults to False.
    bias : bool, optional
        Whether to use a learned bias (with one bias per time step). Defaults to True.
    """

    def __init__(self, feature_dim, seq_len, context=False, bias=True, epsilon=1e-8):
        super(AttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.context = context
        self.bias = bias
        self.epsilon = epsilon

        # Weight vector W of shape (feature_dim,)
        self.W = nn.Parameter(torch.Tensor(feature_dim))

        # Optional bias of shape (seq_len,)
        if bias:
            self.b = nn.Parameter(torch.Tensor(seq_len))
        else:
            self.register_parameter("b", None)

        # Optional context vector u of shape (feature_dim,)
        if context:
            self.u = nn.Parameter(torch.Tensor(feature_dim))
        else:
            self.register_parameter("u", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weights using Xavier (Glorot) uniform initialization.
        nn.init.xavier_uniform_(self.W.unsqueeze(0))
        if self.bias:
            nn.init.zeros_(self.b)
        if self.context:
            nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x, mask=None):
        """
        Forward pass for the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, feature_dim).
        mask : torch.Tensor, optional
            A mask tensor of shape (batch_size, seq_len) with 1 for valid timesteps and 0 for masked timesteps.
            Defaults to None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, feature_dim) which is the weighted sum over time.
        """
        # Compute the attention scores.
        # First, compute the dot product between each time-step input and the weight vector.
        # x: (batch_size, seq_len, feature_dim)
        # self.W: (feature_dim,) -> result is (batch_size, seq_len)
        a = torch.matmul(x, self.W)

        # Add bias if applicable (bias is per time step)
        if self.bias:
            # self.b has shape (seq_len,), and will be broadcast over the batch dimension.
            a = a + self.b

        # Apply tanh nonlinearity.
        a = torch.tanh(a)

        # If context is used, override a with a new dot product with the context vector.
        if self.context:
            # Compute dot product with context vector: result shape (batch_size, seq_len)
            a = torch.matmul(x, self.u)

        # Exponentiate the attention scores.
        a = torch.exp(a)

        # Apply mask if provided.
        if mask is not None:
            # Ensure mask is float so multiplication works as expected.
            a = a * mask.float()

        # Normalize the attention scores over the time dimension.
        # Add a small epsilon to avoid division by zero.
        a_sum = torch.sum(a, dim=1, keepdim=True) + self.epsilon
        a = a / a_sum  # Shape: (batch_size, seq_len)

        # Expand the attention weights for multiplication with x.
        a = a.unsqueeze(-1)  # Now shape: (batch_size, seq_len, 1)

        # Multiply the attention weights with the input and sum over time steps.
        weighted_input = x * a  # (batch_size, seq_len, feature_dim)
        output = torch.sum(weighted_input, dim=1)  # (batch_size, feature_dim)
        return output
