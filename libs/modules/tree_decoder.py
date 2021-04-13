from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
from torch.nn import LSTMCell

from allennlp.common import Registrable
from allennlp.nn import util


class TreeDecoder(torch.nn.Module, Registrable):

    """
    This class abstracts the neural architectures for decoding the encoded states and
    embedded previous step prediction vectors into a new sequence of output vectors.
    The implementations of `DecoderNet` is used by implementations of
    `allennlp.modules.seq2seq_decoders.seq_decoder.SeqDecoder` such as
    `allennlp.modules.seq2seq_decoders.seq_decoder.auto_regressive_seq_decoder.AutoRegressiveSeqDecoder`.
    The outputs of this module would be likely used by `allennlp.modules.seq2seq_decoders.seq_decoder.SeqDecoder`
    to apply the final output feedforward layer and softmax.
    # Parameters
    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of target embeddings. Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    decodes_parallel : `bool`, required
        Defines whether the decoder generates multiple next step predictions at in a single `forward`.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `DecoderNet`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        return self.decoder_output_dim

    def is_lstm(self):
        return False

    def forward(
        self,
        projected_decoder_intput: torch.Tensor,
        previous_state: torch.Tensor,
        num_step: int = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a decoding step, and returns dictionary with decoder hidden state or cache and the decoder output.
        The decoder output is a 3d tensor (group_size, steps_count, decoder_output_dim)
        if `self.decodes_parallel` is True, else it is a 2d tensor with (group_size, decoder_output_dim).
        # Parameters
        previous_steps_predictions : `torch.Tensor`, required
            Embeddings of predictions on previous step.
            Shape: (group_size, steps_count, decoder_output_dim)
        encoder_outputs : `torch.Tensor`, required
            Vectors of all encoder outputs.
            Shape: (group_size, max_input_sequence_length, encoder_output_dim)
        source_mask : `torch.BoolTensor`, required
            This tensor contains mask for each input sequence.
            Shape: (group_size, max_input_sequence_length)
        previous_state : `Dict[str, torch.Tensor]`, required
            previous state of decoder
        # Returns
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
        Tuple of new decoder state and decoder output. Output should be used to generate out sequence elements
        """
        raise NotImplementedError()


@TreeDecoder.register("lstm")
class LstmTreeDecoder(TreeDecoder):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.
    # Parameters
    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(
        self,
        decoder_input_dim: int,
        decoder_output_dim: int,
    ) -> None:

        super().__init__(
            decoder_input_dim=decoder_input_dim,
            decoder_output_dim=decoder_output_dim,
        )

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(
            self.decoder_input_dim, self.decoder_output_dim)

    @overrides
    def forward(
        self,
        projected_decoder_intput: torch.Tensor,
        previous_state: torch.Tensor,
        num_step: int = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        decoder_hidden, decoder_context = previous_state

        child_hidden_states = []
        child_context_states = []
        for _ in range(num_step):
            # shape (decoder_hidden): (batch_size, decoder_output_dim)
            # shape (decoder_context): (batch_size, decoder_output_dim)
            decoder_hidden, decoder_context = self._decoder_cell(
                projected_decoder_intput.float(), (decoder_hidden.float(), decoder_context.float())
            )
            child_hidden_states.append(decoder_hidden)
            child_context_states.append(decoder_context)

        if child_context_states:
            child_hidden_states = torch.stack(
                child_hidden_states).transpose(0, 1)
            child_context_states = torch.stack(
                child_context_states).transpose(0, 1)

        return child_hidden_states, child_context_states

    @overrides
    def is_lstm(self):
        return True


@TreeDecoder.register("gts")
class GTSTreeDecoder(TreeDecoder):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.
    # Parameters
    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__(
        )

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        embedding_size = 300
        hidden_size = 512
        dropout = 0.5

        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(
            hidden_size * 3 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(
            hidden_size * 3 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(
            hidden_size * 3 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(
            hidden_size * 3 + embedding_size, hidden_size)

    @overrides
    def forward(
        self,
        embedded_decoder_intput: torch.Tensor,
        selective_read: torch.Tensor,
        current_context: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        embedded_decoder_intput = self.em_dropout(embedded_decoder_intput)
        selective_read = self.em_dropout(selective_read)
        previous_state = self.em_dropout(previous_state)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(
            torch.cat((previous_state, selective_read, current_context, embedded_decoder_intput), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(
            torch.cat((previous_state, selective_read, current_context, embedded_decoder_intput), 1)))
        r_child = torch.tanh(self.generate_r(
            torch.cat((previous_state, selective_read, current_context, embedded_decoder_intput), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(
            torch.cat((previous_state, selective_read, current_context, embedded_decoder_intput), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g

        #
        child_states = torch.cat(
            (l_child.unsqueeze(-2), r_child.unsqueeze(-2)), -2)

        return child_states
