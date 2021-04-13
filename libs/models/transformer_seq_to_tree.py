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

from allennlp.modules.seq2seq_encoders import PytorchTransformer

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch

from allennlp.modules import LayerNorm
from allennlp.nn.util import sort_batch_by_length
from allennlp.modules.seq2vec_encoders import BertPooler

from libs.modules.tree_decoder import LstmTreeDecoder, GTSTreeDecoder
from libs.modules.state_stack_manager import LSTMStateStackManager, GRUStateStackManager
from libs.modules.other_modules import TreeAttn
from libs.GTS.train_and_evaluate import TreeNode, masked_cross_entropy, TreeEmbedding, generate_tree_input
from libs.GTS.train_and_evaluate import get_all_number_encoder_outputs, TreeBeam, copy_list
from libs.GTS.models import Prediction, GenerateNode, Merge, EncoderSeq, GenerateNode
import copy
from libs.tools.knowledge_graph import create_KGs_with_id_table, formula_to_args
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from libs.modules.GNNs import GNN

USE_CUDA = torch.cuda.is_available()


logger = logging.getLogger(__name__)


@Model.register("transformer_seq2tree")
class TransformerSeq2Tree(Model):
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
        encoder: Seq2SeqEncoder,
        attention: Attention,
        beam_size: int,
        max_decoding_steps: int,
        transformer_model_name: str = "bert-base-chinese",
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

        print(vocab)

        # Target vocabulary and its auxiliary indices
        # The indices that we need during the
        self._target_namespace = target_namespace
        # self._start_index = self.vocab.get_token_index(
        #     START_SYMBOL, self._target_namespace)
        # self._end_index = self.vocab.get_token_index(
        #     END_SYMBOL, self._target_namespace)
        # self._oov_index = self.vocab.get_token_index(
        #     self.vocab._oov_token, self._target_namespace)
        # self._pad_index = self.vocab.get_token_index(
        #     self.vocab._padding_token, self._target_namespace)

        self._target_vocab_size = self.vocab.get_vocab_size(
            self._target_namespace)

        # A list of stacks that store the states for each example in the input data.
        # We dynamically initialize these stacks because the total number of stacks
        # is subject to the batch_size and beam_size.
        # self._state_stacks = []
        # self._state_stack_manager = LSTMStateStackManager()
        self._state_stack_manager = GRUStateStackManager()

        # Encoding modules.
        # self._source_embedder = BasicTextFieldEmbedder(
        #     {"tokens": PretrainedTransformerEmbedder(
        #         transformer_model_name, tokenizer_kwargs={
        #             "additional_special_tokens": ["<NUM>"]})}
        # )
        self._source_embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(num_embeddings=self.vocab.get_vocab_size("tokens"), embedding_dim=512)})

        self._encoder = PytorchTransformer(512, 2)
        self._pooler = BertPooler(
            transformer_model_name,
            dropout=0.1,
        )
        self.proj = nn.Linear(768, 512)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = self._encoder.get_output_dim()//2
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim

        # Decoding modules
        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence that matches the target
        # token from the previous timestep.
        self._target_embedding_dim = self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(
            num_embeddings=self._target_vocab_size, embedding_dim=self._target_embedding_dim
        )
        self._attention = attention
        self._input_projection_layer = Linear(
            self._target_embedding_dim + self.encoder_output_dim * 2, self.decoder_input_dim
        )
        # self._decoder = LstmTreeDecoder(
        # self.decoder_input_dim, self.decoder_output_dim)
        self._decoder = GTSTreeDecoder()
        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(
            self.decoder_output_dim, self._target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(
            self.encoder_output_dim, self.decoder_output_dim)

        # Others

        self.new_id = {
            # need to recover later
            "@@PADDING@@": 0,
            "+": 0,
            "-": 1,
            "*": 2,
            "/": 3,
            "^": 4,
            "square_area": 5,
            "square_perimeter": 6,
            "cubic_volume": 7,
            "circle_area": 8,
            "circumference_radius": 9,
            "circumference_diameter": 10,
            "triangle_area": 11,
            "rectangle_area": 12,
            "rectangle_perimeter": 13,
            "cuboid_volume": 14,
            "cuboid_surface": 15,
            "3.14": 16,
            "1": 17,
            "<N0>": 18,
            "<N1>": 19,
            "<N2>": 20,
            "<N3>": 21,
            "<N4>": 22,
            "<N5>": 23,
            "<N6>": 24,
            "<N7>": 25,
            "@@UNKNOWN@@": 26,
        }
        self.num_start = 16
        self.new_id_back = {v: k for k, v in self.new_id.items()}
        self._target_size = len(self.new_id)
        print(self._target_size)
        # print(self._target_vocab_size)
        # print(self._target_vocab_size)

        self.copy_nums = 8
        self.generate_nums = [3.14, 1]
        self._source_vocab_size = self.vocab.get_vocab_size("tokens")
        embedding_size = 128
        self.hidden_size = hidden_size = 512
        self.encoder = EncoderSeq(input_size=self._source_vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                                  n_layers=2)
        self.predict = Prediction(hidden_size=hidden_size, op_nums=self._target_size - self.copy_nums - 2 - len(self.generate_nums),
                                  input_size=len(self.generate_nums), GNN_out=GNN_out)
        # self.generate = GenerateNode(hidden_size=hidden_size, op_nums=self._target_vocab_size - self.copy_nums - 2 - len(self.generate_nums),
        #                              embedding_size=embedding_size)
        self.generate = GenerateNodeRNN(hidden_size=hidden_size, op_nums=self._target_size - self.copy_nums - 2 - len(self.generate_nums),
                                        embedding_size=embedding_size)
        self.merge = Merge(hidden_size=hidden_size,
                           embedding_size=embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = LayerNorm(1024)

        # We need to know ... to decide when decoding.
        self._number_of_branches = self._initialize_number_of_branch_map(
            number_of_branch_map)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_size = beam_size

    @ overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        # Parameters
        source_tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : `torch.Tensor`, required
            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, source_sequence_length)`.
        source_to_target : `torch.Tensor`, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, source_sequence_length)`.
        metadata : `List[Dict[str, Any]]`, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : `torch.Tensor`, optional (default = `None`)
            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.
        # Returns
        `Dict[str, torch.Tensor]`
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

    def convert_to_new_id(self, x):
        return self.new_id[self.vocab.get_token_from_index(x, self._target_namespace)]

    def get_child_kg_embeddings(self, input_index):
        formula = self.new_id_back[input_index]
        args = formula_to_args[formula]
        embeddings = [self.kg_embedding[x] for x in reversed(args)]
        return embeddings

    def _forward_loss(self, input_batch, target_batch, metadata, state):
        """

        """
        # Prepare the data
        seq_mask = state["source_mask"]
        num_size_batch = [len(x["numbers"]) for x in metadata]
        num_pos = [x["positions"] for x in metadata]
        input_length = seq_mask.sum(-1).cpu()
        target_length = state["target_mask"].sum(-1).cpu()
        nums_stack_batch = self.generate_num_stack(metadata)

        # print(len(nums_stack_batch), len(nums_stack_batch[0]))
        # print(input_batch["tokens"]["tokens"].size())
        # input_batch =

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
        # input_var = input_batch["tokens"]["tokens"].transpose(0, 1)

        # print(self.vocab.get_index_to_token_vocabulary(self._target_namespace))
        # print(self.vocab.get_vocab_size(self._target_namespace))
        target = target_batch["tokens"]["tokens"].transpose(0, 1)
        target = [[self.convert_to_new_id(x.item())
                   for x in y] for y in target]
        target = torch.tensor(
            target, device=target_batch["tokens"]["tokens"].device)
        # print(target.size())

        # 改成 768
        padding_hidden = torch.tensor(
            [0.0 for _ in range(self.predict.hidden_size)], dtype=torch.float, device=target.device).unsqueeze(0)
        batch_size = len(input_length)

        # Run words through encoder
        # 20210117
        # B*S*H, B*H
        encoder_outputs = self._source_embedder(input_batch)
        encoder_outputs = self._encoder(encoder_outputs, seq_mask)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        problem_output = encoder_outputs[0, :, :]
        # encoder_outputs, problem_output = self.encoder(input_var, input_length)
        # encoder_outputs, problem_output = self._encode(input_batch)

        # print(encoder_outputs.size())
        # print(problem_output.size())

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                                  encoder_outputs.size(-1))

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

            # print(generate_input)
            num_of_children = self._get_input_number_of_branches(
                target[t])
            # print(target_t)
            # print(generate_input)
            # print(num_of_children)
            # assert 1 == 2

            child_states, node_label = self.generate(
                current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, child_states_example, node_stack, i, o in zip(range(batch_size), child_states,
                                                                   node_stacks, target[t].tolist(), embeddings_stacks):

                nob = num_of_children[idx]

                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                # Store the states
                if i < self.num_start:
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
                                                          i - self.num_start].unsqueeze(0)
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

                # Prepare left embedding
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        # print("Target", target.size())
        # print("All node outputs", all_node_outputs.size())

        target = target.transpose(0, 1).contiguous()

        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss = masked_cross_entropy(all_node_outputs, target, target_length)

        return {"loss": loss}  # , loss_0.item(), loss_1.item()

    def _forward_prediction(self, input_batch, metadata,
                            beam_size=5, max_length=20) -> Dict[str, torch.Tensor]:

        source_mask = util.get_text_field_mask(input_batch)
        # input_batch = input_batch["tokens"]["tokens"]
        num_poses = [x["positions"] for x in metadata]
        predictions = []
        for input_var, num_pos, meta, seq_mask in zip([input_batch], num_poses, metadata, source_mask):
            # input_var = input_var.unsqueeze(1)
            # input_length, batch_size = input_var.size()

            # input_length = source_mask.sum(-1)

            # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

            num_mask = torch.BoolTensor(
                1, len(num_pos) + len(self.generate_nums)).fill_(0).to(device=source_mask.device)

            padding_hidden = torch.FloatTensor(
                [0.0 for _ in range(self.predict.hidden_size)]).to(device=source_mask.device).unsqueeze(0)

            # Run words through encoder
            # print(input_var.size())
            # encoder_outputs, problem_output = self.encoder(
            #     input_var, [input_length])
            encoder_outputs = self._source_embedder(input_batch)
            encoder_outputs = self._encoder(encoder_outputs, source_mask)
            encoder_outputs = encoder_outputs.transpose(0, 1)
            problem_output = encoder_outputs[0, :, :]

            input_length = encoder_outputs.size(0)
            batch_size = 1
            seq_mask = torch.BoolTensor(1, input_length).fill_(
                0).to(device=source_mask.device)
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

                            num_of_children = self._get_input_number_of_branches(
                                generate_input)
                            nob = num_of_children[0]

                            child_states, node_label = self.generate(
                                current_embeddings, generate_input, current_context)

                            # print(generate_input)
                            # print(nob)
                            # assert nob == 2
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
                                                                  out_token - self.num_start].unsqueeze(0)

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
