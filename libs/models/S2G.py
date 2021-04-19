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


import copy
from libs.tools.knowledge_graph import create_KGs_with_id_table, formula_to_args

from libs.models.seq_to_tree_proto import Seq2Tree
from libs.GTS.train_and_evaluate import get_all_number_encoder_outputs, TreeBeam, copy_list
from libs.GTS.models import Prediction, GenerateNode, Merge, EncoderSeq, GenerateNode
from libs.GTS.train_and_evaluate import TreeNode, masked_cross_entropy, TreeEmbedding, generate_tree_input

from libs.modules.child_generators import BinaryGenerator, GRUGenerator, ChildGenerator


logger = logging.getLogger(__name__)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False, op_type=None):
        self.embedding = embedding
        self.terminal = terminal
        self.op_type = op_type


@Model.register("S2G")
class Seq2GeneralTree(Seq2Tree):
    """
    """

    def __init__(
        self,
        vocab: Vocabulary,
        beam_size: int,
        child_generator: ChildGenerator,
        max_decoding_steps: int,
        target_embedding_dim: int = 30,
        copy_token: str = "@COPY@",
        target_namespace: str = "targe_vocab",
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab, target_namespace)

        # Target vocabulary and its auxiliary indices
        # The indices that we need during the
        self._target_namespace = target_namespace
        self._target_vocab_size = self.vocab.get_vocab_size(
            self._target_namespace)
        self._source_vocab_size = self.vocab.get_vocab_size("tokens")

        embedding_size = 128
        self.hidden_size = hidden_size = 512

        self.encoder = EncoderSeq(input_size=self._source_vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                  n_layers=2)
        self.predict = Prediction(hidden_size=hidden_size, op_nums=self.num_operations,
                                  input_size=self.num_constants)
        # self.generate = GenerateNode(hidden_size=hidden_size, op_nums=self.num_operations,
        #  embedding_size=embedding_size)
        self._target_embedder = Embedding(
            num_embeddings=self.num_operations, embedding_dim=embedding_size
        )
        self.generator = child_generator
        self.merge = Merge(hidden_size=hidden_size,
                           embedding_size=embedding_size)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_size = beam_size

        initializer(self)

    @ overrides
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

    def _forward_loss(self, input_batch, target_batch, metadata, state):
        """

        """
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
        max_num_size = max(num_size_batch) + self.num_constants
        for i in num_size_batch:
            d = i + self.num_constants
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
                target[t].tolist(), outputs, nums_stack_batch, self.num_start_idx, unk)
            target_t = target_t.to(device=target.device)
            generate_input = generate_input.to(device=target.device)
            target[t] = target_t

            current_input = self._target_embedder(generate_input)
            child_states = self.generator(
                current_embeddings, current_context, current_input)
            node_label = current_input

            left_childs = []
            for idx, child_states_example, node_stack, i, o in zip(range(batch_size), child_states,
                                                                   node_stacks, target[t].tolist(), embeddings_stacks):

                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                nob = 2
                # Store the states
                if i < self.num_start_idx:
                    # assert nob == 2
                    for ii, child_state in reversed(list(enumerate(child_states_example[:nob]))):
                        child_state = child_state.unsqueeze(0)
                        # print(child_state.size())
                        if ii == nob-1:
                            node_stack.append(TreeNode(child_state))
                        else:
                            node_stack.append(
                                TreeNode(child_state, left_flag=True))
                    if nob == 2:
                        op_type = "binary"
                    elif nob == 1:
                        op_type = "unary"
                    else:
                        op_type = "ternary"
                    o.append(TreeEmbedding(
                        node_label[idx].unsqueeze(0), False, op_type=op_type))
#                     print(node_label.size())
#                     print(node_label.size())
#                     print(node_label.size())
                else:
                    current_num = current_nums_embeddings[idx,
                                                          i - self.num_start_idx].unsqueeze(0)
#                     print(current_num.size())
#                     print(current_num.size())
#                     print(current_num.size())
                    if len(o) > 0 and o[-1].op_type == "unary":
                        op = o.pop()
                        current_num = self.merge(
                            op.embedding, current_num, current_num)

                    while len(o) > 0 and o[-1].terminal:
                        super_op = None
                        sub_stree = o.pop()
                        op = o.pop()
                        if op.op_type == "ternary":
                            super_op = TreeEmbedding(
                                op.embedding.clone(), False, op_type="ternary")
                        current_num = self.merge(
                            op.embedding, sub_stree.embedding, current_num)

                        if super_op:
                            super_op.op_type = "binary"
                            o.append(super_op)

                    o.append(TreeEmbedding(
                        current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()

        # op_target = target < num_start_idx
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
                1, len(num_pos) + self.num_constants).fill_(0).to(device=input_var.device)

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
                        if out_token < self.num_start_idx:
                            generate_input = torch.LongTensor(
                                [out_token]).to(device=encoder_outputs.device)

                            current_input = self._target_embedder(
                                generate_input)
                            child_states = self.generator(
                                current_embeddings, current_context, current_input)
                            node_label = current_input

                            nob = 2

                            for ii, child_state in reversed(list(enumerate(child_states[0][:nob]))):
                                child_state = child_state.unsqueeze(0)
                                if ii == nob-1:
                                    current_node_stack[0].append(
                                        TreeNode(child_state))
                                else:
                                    current_node_stack[0].append(
                                        TreeNode(child_state, left_flag=True))
                            if nob == 2:
                                op_type = "binary"
                            elif nob == 1:
                                op_type = "unary"
                            else:
                                op_type = "ternary"

                            current_embeddings_stacks[0].append(
                                TreeEmbedding(node_label[0].unsqueeze(0), False, op_type=op_type))
                        else:
                            current_num = current_nums_embeddings[0,
                                                                  out_token - self.num_start_idx].unsqueeze(0)

                            super_op = None
                            if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].op_type == "unary":
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(
                                    op.embedding, current_num, current_num)

                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()

                                if op.op_type == "ternary":
                                    super_op = copy.deepcopy(op)

                                current_num = self.merge(
                                    op.embedding, sub_stree.embedding, current_num)

                                if super_op:
                                    super_op.op_type = "binary"
                                    current_embeddings_stacks[0].append(
                                        super_op)

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
            original_prediction = [self.back_to_old_id(x) for x in prediction]
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
