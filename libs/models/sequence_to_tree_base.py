import logging
from typing import Dict, Tuple, List, Any, Union
from overrides import overrides
import re
import copy

import numpy
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model


logger = logging.getLogger(__name__)


class Seq2Tree(Model):
    """This is the base class for all sequence-to-tree models.

    Here we define the common utility functions for all seq2tree models:

    1. Adjusting the target vocabulary:
        The target vocabulary in Allennlp is randomly ordered. However, we would like tokens in the target vocabulary to have continuous ids
        if they belong to the same type. For example, we want the target vocab to have the form:
            {"+":1, "-":2, "*":3, "/":4, ..., "N1": 10, "<N2>":11, "<N3>":12 },
        where the group of operators (+, -, *, /) and pseudo tokens (<N1>, <N2>, <N3>, ...) all have continuous ids.

        We want to do that because our model predicts a probability distribution for each group. We need to separate different groups by ids
        so that we can design our training flow and integrate the predictions more easily. For instance, we may want to predict a probability
        for class 1~10 (for operators) and predict another probability for class 11~15 (for pesudo tokens). It's difficult to count the total
        number of classes or integrate two probability distributions if the ids of operators and psuedo tokens are scattered across each other.

        To this end, here we provide four utility funcions for rearranging the vocabulary and converting old ids to new ids, and vice versa.

            + `get_vocab_groups()`: Get the tokens separated in different groups (i.e., operators, constants, pseudo tokens).
            + `rearrange_target_vocab()`: It will rearrange the target vocabulary so that tokens of the same type will have continuous ids.
            + `convert_to_new_id()`: Convert the old id of the target token to the new id.
            + `back_to_old_id()`: Convert the new id of the target token to the old id.

        We also define the AllenNLP utility function `make_output_human_readable()` here, since it will use the above
        functions to convert the model predictions (ids) to their original tokens. 

    2. Prepare the positions of the candidate numbers for copying:
        For the numbers that appear only once in the problem text. We replace them with corresponding pseudo tokens 
        and train our model to predict them in the equation, such as predicting "<N0> + <N1>" for "3 + 2".
        However, for numbers that appear more than one time. We do not replace them with pseudo tokens but would collect 
        them in a candidate list and use copy mechanism to select them directly. Hence, we need to prepare the positions of 
        the candidate numbers for copying. Details please refer to `get_copy_positions()`.


    Args:
        vocab : `Vocabulary`, required
            Vocabulary containing source and target vocabularies.
        target_namespace : `str`, optional (default = `'target_tokens'`)
            The namespace for the target vocabulary.

    """

    def __init__(
        self,
        vocab: Vocabulary,
        target_namespace: str = "equation_vocab"
    ) -> None:

        super().__init__(vocab)
        self._target_namespace = target_namespace

        # Rearrange the target vocab
        self.ops, self.constants, self.pseudo_tokens = self.get_vocab_groups()
        self.token_to_new_id = self.rearrange_target_vocab()
        self._target_size = len(self.token_to_new_id)
        self.new_id_to_token = {
            v: k for k, v in self.token_to_new_id.items()}

        # Some numbers
        self.num_operations = len(self.ops)
        self.num_constants = len(self.constants)
        self.num_pseudo_tokens = len(self.pseudo_tokens)

        # The is not a statistics number. This is first id of the constant numbers and pseudo tokens,
        # which follows immediately after the ids of operators and formulas.
        # We need to know this id to do offset.
        self.num_start_id = len(self.ops)

        # Unknown id
        self.unk_id = self.token_to_new_id["@@UNKNOWN@@"]

    def get_vocab_groups(self):
        """
        Get the operator tokens, constant number tokens, and the pseudo tokens (e.g. <N0>, <N1>, ...).    
        Here, we consider formulas as a kind of operators.
        """
        # All target tokens
        vocab_size = self.vocab.get_vocab_size(self._target_namespace)
        tokens = [self.vocab.get_token_from_index(x, self._target_namespace)
                  for x in range(vocab_size)]

        # Operators
        operators = []
        for token in tokens:
            if token in ['+', '-', '*', '/', '^']:
                operators.append(token)

        # Formulas
        formulas = []
        for token in tokens:
            if re.match('^[a-z_]*$', token):
                formulas.append(token)

        operators += formulas

        # Constant numbers
        constants = []
        for token in tokens:
            if re.match('^[0-9\.]*$', token):
                constants.append(token)

        # Pseudo tokens
        pseudo_tokens = []
        for token in tokens:
            if re.match('^<N[0-9]*>$', token):
                pseudo_tokens.append(token)
        pseudo_tokens = sorted(pseudo_tokens)

        return operators, constants, pseudo_tokens

    def rearrange_target_vocab(self):
        """
        We assign new ids to the operators, constant numbers, and pseudo tokens in incremental order.

        Returns:
            token_to_new_id: `dict[str, int]`
                A dict that maps a target token to its new id.

        """
        # Set new vocab mapping
        token_to_new_id = {}

        # Always set padding to zero
        token_to_new_id["@@PADDING@@"] = 0

        # Operators and formulas
        for token in self.ops:
            token_to_new_id[token] = len(token_to_new_id) - 1

        # Constants
        for token in self.constants:
            token_to_new_id[token] = len(token_to_new_id) - 1

        # Pseudo tokens
        for token in self.pseudo_tokens:
            token_to_new_id[token] = len(token_to_new_id) - 1

        # Append unknown to the last
        token_to_new_id["@@UNKNOWN@@"] = len(token_to_new_id) - 1

        return token_to_new_id

    def convert_to_new_id(self, old_id):
        """
        Convert the old id of the target token to the new id.
        """
        token = self.vocab.get_token_from_index(old_id, self._target_namespace)
        new_id = self.token_to_new_id[token]
        return new_id

    def back_to_old_id(self, new_id):
        """
        Convert the new id of the target token to the old id.
        """
        token = self.new_id_to_token[new_id]
        old_id = self.vocab.get_token_index(token, self._target_namespace)
        return old_id

    @ overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Convert the predictions (new ids) to their original ids and tokens.
        """

        batch_original_ids = []
        batch_original_tokens = []

        batch_prediction = output_dict["prediction"]
        for prediction in batch_prediction:

            # We take the top1 candidate of the beam search result.
            predicted_ids = prediction[0]

            # Convert the predicted new ids to original old ids
            original_ids = [self.back_to_old_id(x) for x in predicted_ids]
            original_tokens = [self.vocab.get_token_from_index(
                index, self._target_namespace) for index in original_ids]

            # Store the results
            batch_original_ids.append(original_ids)
            batch_original_tokens.append(original_tokens)

        output_dict["predicted_ids"] = batch_original_ids
        output_dict["predicted_tokens"] = batch_original_tokens
        return output_dict

    # def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """
    #     Not used now.
    #     """
    #     # shape: (batch_size, source_sequence_length, encoder_input_dim)
    #     embedded_input = self._source_embedder(source_tokens)
    #     embedded_input = self.dropout(embedded_input)
    #     # shape: (batch_size, source_sequence_length)
    #     source_mask = util.get_text_field_mask(source_tokens)
    #     # shape: (batch_size, source_sequence_length, encoder_output_dim)
    #     encoder_outputs = self._encoder(embedded_input, source_mask)

    #     final_encoder_output = util.get_final_encoder_states(
    #         encoder_outputs, source_mask, self._encoder.is_bidirectional()
    #     )

    #     problem_output = encoder_outputs[:, -1, :self.hidden_size] + \
    #         encoder_outputs[:, 0, self.hidden_size:]
    #     encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + \
    #         encoder_outputs[:, :, self.hidden_size:]  # S x B x H

    #     return encoder_outputs.transpose(0, 1), problem_output
    #     # return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def get_copy_positions(self, batch_metadata):
        """
        Get the position of the candidate numbers for copying.
        This function is adapted from: https://github.com/ShichaoSun/math_seq2tree/blob/master/src/pre_data.py
        """

        batch_copy_positions = []

        for metadata in batch_metadata:

            copy_positions = []
            for token in metadata["target_tokens"]:
                temp_num = []
                flag_not = True
                if (self.vocab.get_token_index(token, self._target_namespace)
                        == self.vocab.get_token_index("@@UNKNOWN@@", self._target_namespace)):
                    flag_not = False
                    for i, j in enumerate(metadata["numbers"]):
                        if j == token:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    copy_positions.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    copy_positions.append(
                        [_ for _ in range(len(metadata["numbers"]))])
            copy_positions.reverse()

            batch_copy_positions.append(copy_positions)

        return batch_copy_positions
