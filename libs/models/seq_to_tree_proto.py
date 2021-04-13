"""
Implement the base class
"""


"""
Most of the code is adpated from:
https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/copynet_seq2seq.py
"""


import logging
from typing import Dict, Tuple, List, Any, Union
import json
from overrides import overrides
import copy
import numpy
import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
logger = logging.getLogger(__name__)


class Seq2Tree(Model):
    """The base class for seq2tree models.

    We define the common utility functions here so that all the seq2tree models can use them.

    Args:
        vocab : `Vocabulary`, required
            Vocabulary containing source and target vocabularies.
        target_namespace : `str`, optional (default = `'target_tokens'`)
            The namespace for the target vocabulary.

    """

    def __init__(
        self,
        vocab: Vocabulary,
        target_namespace: str = "target_tokens"
    ) -> None:
        super().__init__(vocab)

        self._target_namespace = target_namespace

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        We need numbers and number positions 
        They shouldn't be put in the metadata.

        """
        raise NotImplementedError

    def _forward_loss(self,):

        raise NotImplementedError

    def _forward_prediction(self):

        raise NotImplementedError

    @staticmethod
    def get_copy_positions(self, target_tokens, numbers):
        """

        """
        UNK_ID = self.vocab.get_token_index(
            "@@UNKNOWN@@", self._target_namespace)

        num_stack_batch = []
        for prob_metadata in metadata:
            num_stack = []
            for word in prob_metadata["target_tokens"]:
                temp_num = []
                flag_not = True
                if (self.vocab.get_token_index(word, self._target_namespace)
                        == UNK_ID):
                    flag_not = False
                    for i, j in enumerate(prob_metadata["numbers"]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append(
                        [_ for _ in range(len(prob_metadata["numbers"]))])
            num_stack.reverse()
            num_stack_batch.append(num_stack)
        return num_stack_batch
