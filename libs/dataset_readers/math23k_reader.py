"""
Some of the code is adapted from: 
https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/dataset_readers/copynet_seq2seq.py

TODO
    1. Add parameters
        1. problem special tokens:
            NUM
            N0
            None
        2. prefix or infix
        3. args of equation tokenization

可以讀 original equation

"""

import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable, Union
from overrides import overrides

import numpy as np

from allennlp.data import Token, Instance, DatasetReader
from allennlp.data.fields import MetadataField, TextField, SpanField, NamespaceSwappingField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from libs.dataset_readers.text import preprocess_text
from libs.dataset_readers.equation import preprocess_equation
from libs.dataset_readers.equation_utils import convert_expression_notation

logger = logging.getLogger(__name__)


@DatasetReader.register("math23k")
class Math23kReader(DatasetReader):
    """

    """

    def __init__(
        self,
        num_token_type: Union[str, None] = None,
        occurrence_limit: bool = True,
        target_notation: str = "prefix",
        target_namespace: str = "equation_vocab",
        use_original_equation: bool = True,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.num_token_type = num_token_type  # "Num", "Nx" or None
        self.occurrence_limit = occurrence_limit
        self.target_notation = target_notation  # Prefix, infix, postfix
        self._target_namespace = target_namespace
        self._use_original_equation = use_original_equation

        self._source_token_indexers = {
            "tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        with open_compressed(cached_path(file_path), "r") as data_file:
            dataset = json.load(data_file)
            logger.info(
                "Reading instances from file at: %s", file_path)

            for problem in dataset:
                qid = problem["id"]
                text = problem["segmented_text"]
                if self._use_original_equation and "history" in problem and len(problem["history"]) > 0:
                    equation = problem["history"][0]
                else:
                    equation = problem["equation"]
                answer = problem["ans"]
                yield self.text_to_instance(qid, text, equation, answer)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        text: str,
        equation: str,
        answer: str,
    ) -> Instance:

        # Prepare the problem input
        problem_tokens, numbers, positions = preprocess_text(
            text, self.num_token_type)
        tokens = [Token(x) for x in problem_tokens]
        problem_field = TextField(
            tokens,
            token_indexers=self._source_token_indexers,
        )

        # Prepare the equation output
        expression = equation[2:]  # x=a+b+c -> a+b+c
        expression_tokens = preprocess_equation(
            expression, numbers, self.occurrence_limit)
        expression_tokens = convert_expression_notation(
            expression_tokens, self.target_notation)
        tokens = [Token(x) for x in expression_tokens]
        equation_field = TextField(
            tokens,
            token_indexers=self._target_token_indexers,
        )

        # Wrap them all
        metadata = {
            "id": qid,
            "problem": text,
            "equation": equation,
            "answer": answer,
            "numbers": numbers,
            "positions": positions,
            "source_tokens": problem_tokens,
            "target_tokens": expression_tokens
        }

        fields = {
            "source_tokens": problem_field,
            "target_tokens": equation_field,
            "metadata": MetadataField(metadata)
        }

        return Instance(fields)


# def get
