"""
Most of the code is adpated from:
https://github.com/allenai/allennlp-models/blob/main/allennlp_models/generation/models/copynet_seq2seq.py
"""

import logging
from typing import Dict, Tuple, List, Any, Union
import json

import numpy
from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch

from allennlp.modules import LayerNorm
from allennlp.nn.util import sort_batch_by_length

from libs.modules.tree_decoder import LstmTreeDecoder, GTSTreeDecoder
from libs.modules.state_stack_manager import LSTMStateStackManager, GRUStateStackManager
from libs.modules.other_modules import TreeAttn
import copy
from libs.tools.knowledge_graph import create_KGs_with_id_table, formula_to_args
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from libs.modules.GNNs import GNN


from libs.GTS.train_and_evaluate import get_all_number_encoder_outputs, TreeBeam, copy_list
from libs.GTS.models import Prediction, GenerateNode, Merge, EncoderSeq, GenerateNode
from libs.GTS.train_and_evaluate import TreeNode, masked_cross_entropy, TreeEmbedding, generate_tree_input


logger = logging.getLogger(__name__)


@Model.register("seq2tree")
class Seq2Tree(Model):
    """
    This is an implementation of [CopyNet](https://arxiv.org/pdf/1603.06393).
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.
    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.
    # Parameters
    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    attention : `Attention`, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : `int`, required
        Beam width to use for beam search prediction.
    max_decoding_steps : `int`, required
        Maximum sequence length of target predictions.
    target_embedding_dim : `int`, optional (default = `30`)
        The size of the embeddings for the target vocabulary.
    copy_token : `str`, optional (default = `'@COPY@'`)
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    target_namespace : `str`, optional (default = `'target_tokens'`)
        The namespace for the target vocabulary.
    tensor_based_metric : `Metric`, optional (default = `'BLEU'`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    initializer : `InitializerApplicator`, optional
        An initialization strategy for the model weights.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        number_of_branch_map: Dict[str, int],
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        beam_size: int,
        max_decoding_steps: int,
        target_embedding_dim: int = 30,
        copy_token: str = "@COPY@",
        target_namespace: str = "targe_vocab",
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        GNN_dim: int = 512,
        GNN_out: int = 512,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab)

        # Target vocabulary and its auxiliary indices
        # The indices that we need during the
        self._target_namespace = target_namespace
        self._target_vocab_size = self.vocab.get_vocab_size(
            self._target_namespace)

        # A list of stacks that store the states for each example in the input data.
        # We dynamically initialize these stacks because the total number of stacks
        # is subject to the batch_size and beam_size.
        # self._state_stacks = []
        # self._state_stack_manager = LSTMStateStackManager()
        # self._state_stack_manager = GRUStateStackManager()

        # Others

        self.new_id = {
            # need to recover later
            "@@PADDING@@": 0,
            "*": 0,
            "/": 1,
            "+": 2,
            "-": 3,
            "^": 4,
            "3.14": 5,
            "1": 6,
            "<N0>": 7,
            "<N1>": 8,
            "<N2>": 9,
            "<N3>": 10,
            "<N4>": 11,
            "<N5>": 12,
            "<N6>": 13,
            "<N7>": 14,
            "@@UNKNOWN@@": 15
        }
        self.num_start = 5
        self.new_id_back = {v: k for k, v in self.new_id.items()}
        self._target_size = len(self.new_id)

        self.copy_nums = 8
        self.generate_nums = [3.14, 1]
        embedding_size = 128
        self.hidden_size = hidden_size = 512
        self._source_vocab_size = self.vocab.get_vocab_size("tokens")

        self.encoder = EncoderSeq(input_size=self._source_vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                  n_layers=2)
        self.predict = Prediction(hidden_size=hidden_size, op_nums=self._target_size - self.copy_nums - 2 - len(self.generate_nums),
                                  input_size=len(self.generate_nums))
        self.generate = GenerateNode(hidden_size=hidden_size, op_nums=self._target_size - self.copy_nums - 2 - len(self.generate_nums),
                                     embedding_size=embedding_size)
        self.merge = Merge(hidden_size=hidden_size,
                           embedding_size=embedding_size)

        # We need to know ... to decide when decoding.
        self._number_of_branches = self._initialize_number_of_branch_map(
            number_of_branch_map)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_size = beam_size

#         initializer(self)

        
        
    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        """

        state = {
            "source_mask": util.get_text_field_mask(source_tokens),
            "target_mask": util.get_text_field_mask(target_tokens),
        }

        if target_tokens:
            output_dict = self._forward_loss(
                source_tokens, target_tokens, metadata, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata
        if target_tokens:
            output_dict["target_tokens"] = target_tokens["tokens"]["tokens"]

        if not self.training:
            predictions = self._forward_prediction(source_tokens, metadata)
            output_dict.update(predictions)

        return output_dict

    def generate_num_stack(self, metadata):

        num_stack_batch = []
        for prob_metadata in metadata:
            num_stack = []
            for word in prob_metadata["target_tokens"]:
                temp_num = []
                flag_not = True
                if (self.vocab.get_token_index(word, self._target_namespace)
                        == self.vocab.get_token_index("@@UNKNOWN@@", self._target_namespace)):
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

    def convert_to_new_id(self, x):
        return self.new_id[self.vocab.get_token_from_index(x, self._target_namespace)]

    def _forward_loss(self, input_batch, target_batch, metadata, state):
        """

        """
        num_start = self.num_start
        num_size_batch = [len(x["numbers"]) for x in metadata]
        num_pos = [x["positions"] for x in metadata]
        nums_stack_batch = self.generate_num_stack(metadata)

        # Prepare the data
        input_length = state["source_mask"].sum(-1).cpu()
        target_length = state["target_mask"].sum(-1).cpu()
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] +
                            [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask).to(
            device=state["source_mask"].device)

        num_mask = []
        max_num_size = max(num_size_batch) + len(self.generate_nums)
        for i in num_size_batch:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.tensor(
            num_mask, dtype=torch.bool, device=seq_mask.device)

        unk = self.convert_to_new_id(self.vocab.get_token_index(
            "@@UNKNOWN@@", self._target_namespace))

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch["tokens"]["tokens"].transpose(0, 1)
        target = target_batch["tokens"]["tokens"].transpose(0, 1)
        target = [[self.convert_to_new_id(x.item())
                   for x in y] for y in target]
        target = torch.tensor(target, device=input_var.device)

        padding_hidden = torch.tensor(
            [0.0 for _ in range(self.predict.hidden_size)], dtype=torch.float, device=target.device).unsqueeze(0)
        batch_size = len(input_length)

        # S*B*H,B*H
        encoder_outputs, problem_output = self.encoder(input_var, input_length)

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                  self.encoder.hidden_size)

        # num_start = output_lang.num_start
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        for t in range(max_target_length):

            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # 64, 4, 512; 64, 6, 512
            # print(all_nums_encoder_outputs.size())
            # print(current_nums_embeddings.size())

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = generate_tree_input(
                target[t].tolist(), outputs, nums_stack_batch, self.num_start, unk)
            target_t = target_t.to(device=target.device)
            generate_input = generate_input.to(device=target.device)
            target[t] = target_t

            left_child, right_child, node_label = self.generate(
                current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t].tolist(), embeddings_stacks):

                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                # Store the states
                if i < self.num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(
                        node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx,
                                                          i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(
                            op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()

        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss = masked_cross_entropy(all_node_outputs, target, target_length)

        return {"loss": loss}  # , loss_0.item(), loss_1.item()

    def _forward_prediction(self, input_batch, metadata,
                            beam_size=5, max_length=20) -> Dict[str, torch.Tensor]:

        source_mask = util.get_text_field_mask(input_batch)
        input_batch = input_batch["tokens"]["tokens"]
        num_poses = [x["positions"] for x in metadata]

        predictions = []
        for input_var, num_pos, meta, seq_mask in zip(input_batch, num_poses, metadata, source_mask):

            input_var = input_var.unsqueeze(1)
            input_length, batch_size = input_var.size()
            seq_mask = torch.BoolTensor(1, input_length).fill_(
                0).to(device=input_var.device)

            # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

            num_mask = torch.BoolTensor(
                1, len(num_pos) + len(self.generate_nums)).fill_(0).to(device=input_var.device)

            padding_hidden = torch.FloatTensor(
                [0.0 for _ in range(self.predict.hidden_size)]).to(device=input_var.device).unsqueeze(0)

            # Run words through encoder
            # print(input_var.size())
            encoder_outputs, problem_output = self.encoder(
                input_var, [input_length])
            # encoder_outputs, problem_output = self._encode(
            # {"tokens": {"tokens": input_var.transpose(0, 1)}})

            # Prepare input and output variables
            node_stacks = [[TreeNode(_)]
                           for _ in problem_output.split(1, dim=0)]

            num_size = len(num_pos)
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                                      self.encoder.hidden_size)

            # B x P x N
            embeddings_stacks = [[] for _ in range(batch_size)]
            left_childs = [None for _ in range(batch_size)]

            beams = [
                TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

            for t in range(max_length):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    # left_childs = torch.stack(b.left_childs)
                    left_childs = b.left_childs

                    num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                        b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                        seq_mask, num_mask)

                    out_score = nn.functional.log_softmax(
                        torch.cat((op, num_score), dim=1), dim=1)

                    topv, topi = out_score.topk(beam_size)

                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(
                            b.embedding_stack)
                        current_out = copy.deepcopy(b.out)

                        out_token = int(ti)
                        current_out.append(out_token)

                        node = current_node_stack[0].pop()

                        # Store states
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor(
                                [out_token]).to(device=encoder_outputs.device)

                            left_child, right_child, node_label = self.generate(
                                current_embeddings, generate_input, current_context)

                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(
                                TreeNode(left_child, left_flag=True))

                            current_embeddings_stacks[0].append(
                                TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0,
                                                                  out_token - self.num_start].unsqueeze(0)

                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(
                                    op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(
                                TreeEmbedding(current_num, True))

                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(
                                current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                      current_left_childs, current_out))
                beams = sorted(
                    current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            predictions.append([beams[0].out])
        return {"predictions": predictions}

    def _initialize_number_of_branch_map(self, number_of_branch_map):
        """

        """
        #
        number_of_branches = torch.zeros(
            self._target_size, 1)

        for node_type, num in number_of_branch_map.items():
            index = self.new_id[node_type]
            number_of_branches[index] = num

        return number_of_branches

    def _get_input_number_of_branches(self, input_indices):
        """
        """
        return torch.index_select(
            self._number_of_branches.to(
                device=input_indices.device), 0, input_indices
        ).contiguous().int()

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, source_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        embedded_input = self.dropout(embedded_input)
        # shape: (batch_size, source_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, source_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs, source_mask, self._encoder.is_bidirectional()
        )


#         encoder_outputs = self.layer_norm(encoder_outputs)

        problem_output = encoder_outputs[:, -1, :self.hidden_size] + \
            encoder_outputs[:, 0, self.hidden_size:]
        # print(problem_output.size())
        # problem_output = final_encoder_output[:, :self.hidden_size] + \
        #     final_encoder_output[:, self.hidden_size:]
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + \
            encoder_outputs[:, :, self.hidden_size:]  # S x B x H

        # print(problem_output.size())
        # print(encoder_outputs.size())
        return encoder_outputs.transpose(0, 1), problem_output
        # return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    @ overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.
        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """

        predicted_tokens = []
        original_predictions = []
        for prediction in output_dict["predictions"]:
            prediction = prediction[0]
            original_prediction = [self.new_id_back[x] for x in prediction]
            original_prediction = [self.vocab.get_token_index(
                x, self._target_namespace) for x in original_prediction]
            original_predictions.append(original_prediction)
            tokens = [self.vocab.get_token_from_index(
                index, self._target_namespace) for index in original_prediction]
            predicted_tokens.append(tokens)
        output_dict["predicted_tokens"] = predicted_tokens
        output_dict["predictions"] = original_predictions
        return output_dict

    # @ overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     all_metrics: Dict[str, float] = {}
    #     if not self.training:
    #         if self._tensor_based_metric is not None:
    #             all_metrics.update(
    #                 self._tensor_based_metric.get_metric(
    #                     reset=reset)  # type: ignore
    #             )
    #         if self._token_based_metric is not None:
    #             all_metrics.update(self._token_based_metric.get_metric(
    #                 reset=reset))  # type: ignore
    #     return all_metrics
