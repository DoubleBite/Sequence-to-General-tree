"""

Part of the model flow is adapted from:https://github.com/ShichaoSun/math_seq2tree
"""

import logging
from typing import Dict, Tuple, List, Any, Union
from collections import defaultdict
import copy

# Numpy and torch
import numpy
import torch
import torch.nn as nn

# Allennlp classes and functions
from overrides import overrides
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util

# Self-defined packages
from libs.models.sequence_to_tree_base import Seq2Tree
from libs.modules.gts_modules import Prediction, Merge, EncoderSeq
from libs.modules.child_node_generators import ChildNodeGenerator
from libs.tools.gts_utils import TreeNode, masked_cross_entropy, TreeEmbedding, generate_tree_input
from libs.tools.gts_utils import get_all_number_encoder_outputs, TreeBeam, copy_list


logger = logging.getLogger(__name__)


@Model.register("S2G-test")
class Seq2GeneralTreeTest(Seq2Tree):
    """
    Some descriptions

    Args:
        number_of_branch_map: Dict[str, int], `required`
            Mapping between the operators/formulas and their corresponding number of children/branches. 

    """

    def __init__(
        self,
        vocab: Vocabulary,
        number_of_branch_map: Dict[str, int],
        child_node_generator: ChildNodeGenerator,
        target_namespace: str = "equation_vocab",
        embedding_size: int = 128,
        hidden_size: int = 512,
        beam_size: int = 5,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab, target_namespace)

        # Since nodes could have arbitrary number of children. We need have a map to look it up.
        self.number_of_branch_map = number_of_branch_map

        # GTS modules
        self._source_vocab_size = self.vocab.get_vocab_size("tokens")
        self.encoder = EncoderSeq(input_size=self._source_vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                  n_layers=2)
        self.predict = Prediction(hidden_size=hidden_size, op_nums=self.num_operations,
                                  input_size=self.num_constants)
        self._target_embedder = Embedding(
            num_embeddings=self.num_operations, embedding_dim=embedding_size)
        self.merge = Merge(hidden_size=hidden_size,
                           embedding_size=embedding_size)

        # The generator to generate arbitrary number of child states
        self.generator = child_node_generator

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

        output_dict = {}

        if target_tokens:
            loss = self._forward_loss(
                source_tokens, target_tokens, metadata)
            output_dict.update(loss)

        if not self.training:
            prediction = self._forward_prediction(
                source_tokens, metadata)
            output_dict.update(prediction)

        return output_dict

    def _forward_loss(self, source_batch, target_batch, batch_metadata):
        """

        """

        # Get the data from allennlp wrappers and transpose to sequence first.
        # Also, convert target tokens to new ids
        source_tokens = source_batch["tokens"]["tokens"].transpose(0, 1)
        target_tokens = target_batch["tokens"]["tokens"].transpose(0, 1)
        tmp = [[self.convert_to_new_id(x.item())
                for x in y] for y in target_tokens]
        target_tokens = torch.tensor(tmp, device=source_tokens.device)

        # Get the masks and lengths.
        source_mask = util.get_text_field_mask(source_batch)
        target_mask = util.get_text_field_mask(target_batch)
        source_mask_inverse = ~source_mask
        source_length = source_mask.sum(-1).cpu()
        target_length = target_mask.sum(-1).cpu()
        max_target_length = max(target_length)

        # Other metadata
        number_positions = [metadata["positions"]
                            for metadata in batch_metadata]
        copy_positions = self.get_copy_positions(batch_metadata)

        # Generate number mask
        number_sizes = [len(metadata["numbers"])
                        for metadata in batch_metadata]
        max_num_size = max(number_sizes)
        num_mask = []
        for size in number_sizes:
            num_mask.append([0] * (size + self.num_constants) +
                            [1] * (max_num_size - size))
        num_mask = torch.tensor(
            num_mask, dtype=torch.bool, device=source_mask.device)

        # Encode source tokens
        # encoder_outputs: seq_length * batch_size * hidden_size
        # problem_output: batch_size * hidden_size
        encoder_outputs, problem_output = self.encoder(
            source_tokens, source_length)
        seq_length, batch_size, hidden_size = encoder_outputs.size()

        # Get the representations of the numbers in the problem text
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, number_positions, batch_size, max_num_size,
                                                                  hidden_size)

        # Prepare containers for tree generation
        all_node_outputs = []
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        padding_hidden = encoder_outputs.new_zeros(1, hidden_size)

        # Start tree generation
        for t in range(max_target_length):

            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, source_mask_inverse, num_mask)

            # Prediction for op and pseudo tokens
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            #
            target_t, generate_input = generate_tree_input(
                target_tokens[t].tolist(), outputs, copy_positions, self.num_start_id, self.unk_id)
            target_t = target_t.to(device=target_tokens.device)
            generate_input = generate_input.to(device=target_tokens.device)
            target_tokens[t] = target_t

            # Generate child states
            current_input = self._target_embedder(generate_input)
            batch_child_states = self.generator(
                current_embeddings, current_context, current_input)
            node_label = current_input

            # We need to walk through the tree states of each example in the batch
            left_childs = []
            for idx, child_states, node_stack, i, o in zip(range(batch_size), batch_child_states,
                                                           node_stacks, target_tokens[t].tolist(), embeddings_stacks):

                # If there is nodes in the stack, we will continue generation.
                # If the stack is empty, then the tree generation is over. Just wait for other trees in the batch.
                if len(node_stack) != 0:
                    node_stack.pop()
                else:

                    left_childs.append(None)
                    continue

                # If the node is an operator or formula
                if i < self.num_start_id:

                    # Different nodes have different number of children. We need to look it up.
                    num_of_children = self._get_number_of_children(i)
                    child_states = child_states[:num_of_children]

                    # Append the child states to the stack
                    for ii, child_state in enumerate(reversed(child_states)):
                        child_state = child_state.unsqueeze(0)
                        if ii == num_of_children-1:  # The last child
                            node_stack.append(TreeNode(child_state))
                        else:
                            node_stack.append(
                                TreeNode(child_state, left_flag=True))

                    if num_of_children == 2:
                        op_type = "binary"
                    elif num_of_children == 1:
                        op_type = "unary"
                    else:
                        op_type = "ternary"
                    o.append(TreeEmbedding(
                        node_label[idx].unsqueeze(0), False, op_type=op_type))

                # If the node is a number
                else:
                    current_num = current_nums_embeddings[idx,
                                                          i - self.num_start_id].unsqueeze(0)
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

                # If there is a left sibling
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # Calculate the loss
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        target = target_tokens.transpose(0, 1).contiguous()
        loss = masked_cross_entropy(all_node_outputs, target, target_length)

        return {"loss": loss}

    def _forward_prediction(self, source_batch, batch_metadata,
                            max_length=20) -> Dict[str, torch.Tensor]:

        # Here we predict each problem one by one.
        # So we use the name `batch_xxx` to indicate that we're gonna iterate through it.
        batch_source_tokens = source_batch["tokens"]["tokens"]
        batch_number_positions = [metadata["positions"]
                                  for metadata in batch_metadata]

        prediction = []
        for source_tokens, num_pos in zip(batch_source_tokens, batch_number_positions):

            source_tokens = source_tokens.unsqueeze(1)
            source_length, batch_size = source_tokens.size()
            num_size = len(num_pos)

            # Create the masks
            source_mask = source_tokens.new_zeros(
                1, source_length,  dtype=torch.bool)
            num_mask = source_tokens.new_zeros(
                1, len(num_pos) + self.num_constants,  dtype=torch.bool)

            # Run words through encoder
            encoder_outputs, problem_output = self.encoder(
                source_tokens, [source_length])
            _, _, hidden_size = encoder_outputs.size()
            padding_hidden = encoder_outputs.new_zeros(1, hidden_size)

            # Get the representations of the numbers in the problem text
            all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                                      hidden_size)

            # Prepare containers for tree generation
            node_stacks = [[TreeNode(_)]
                           for _ in problem_output.split(1, dim=0)]
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
                        source_mask, num_mask)

                    out_score = nn.functional.log_softmax(
                        torch.cat((op, num_score), dim=1), dim=1)

                    topv, topi = out_score.topk(self._beam_size)

                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(
                            b.embedding_stack)
                        current_out = copy.deepcopy(b.out)

                        out_token = int(ti)
                        current_out.append(out_token)

                        node = current_node_stack[0].pop()

                        # If the node is an operator or formula
                        if out_token < self.num_start_id:

                            generate_input = torch.LongTensor(
                                [out_token]).to(device=encoder_outputs.device)
                            current_input = self._target_embedder(
                                generate_input)
                            child_states = self.generator(
                                current_embeddings, current_context, current_input)
                            node_label = current_input

                            # Different nodes have different number of children. We need to look it up.
                            num_of_children = self._get_number_of_children(
                                out_token)
                            child_states = child_states[0][:num_of_children]

                            for ii, child_state in enumerate(reversed(child_states)):
                                child_state = child_state.unsqueeze(0)
                                if ii == num_of_children-1:
                                    current_node_stack[0].append(
                                        TreeNode(child_state))
                                else:
                                    current_node_stack[0].append(
                                        TreeNode(child_state, left_flag=True))
                            if num_of_children == 2:
                                op_type = "binary"
                            elif num_of_children == 1:
                                op_type = "unary"
                            else:
                                op_type = "ternary"

                            current_embeddings_stacks[0].append(
                                TreeEmbedding(node_label[0].unsqueeze(0), False, op_type=op_type))
                        else:
                            current_num = current_nums_embeddings[0,
                                                                  out_token - self.num_start_id].unsqueeze(0)

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
                beams = beams[:self._beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            prediction.append([beams[0].out])
        return {"prediction": prediction}

    # def _initialize_number_of_branch_map(self, number_of_branch_map):
    #     """

    #     """
    #     #
    #     number_of_branches = torch.zeros(
    #         self._target_size, 1)

    #     for node_type, num in number_of_branch_map.items():
    #         if node_type in self.token_to_new_id:
    #             index = self.token_to_new_id[node_type]
    #             number_of_branches[index] = num

    #     return number_of_branches

    # def _get_input_number_of_branches(self, input_indices):
    #     """
    #     """
    #     # return [self.number_of_branch_map[self.new_id_to_token[x.item()]] for x in input_indices]
    #     return torch.index_select(
    #         self._number_of_branches.to(
    #             device=input_indices.device), 0, input_indices
    #     ).contiguous().int()

    def _get_number_of_children(self, node_id):
        """
        """
        token = self.new_id_to_token[node_id]
        return self.number_of_branch_map[token]
