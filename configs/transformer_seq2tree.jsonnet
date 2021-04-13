#
local equation_vocab_str = importstr '../extra_files/equation_vocab.txt';
local equation_vocab = std.split(equation_vocab_str, "\n");
#
local number_of_branches = import '../extra_files/number_of_branches.jsonnet';

{
    "train_data_path": "data/geometry_5fold/fold0_train.json",
    //"validation_data_path": "data/geometry_5fold/fold0_test.json",
    "dataset_reader": {
        "type": "math23k",
        "num_token_type": "NUM",
    },
    "vocabulary":{
       "pretrained_files": {
           "target_vocab": "extra_files/equation_vocab.txt",
       },
       "only_include_pretrained_words": true,
    },
    "model": {
        "type": "transformer_seq2tree",
        "target_namespace": "target_vocab",
        "number_of_branch_map": number_of_branches,
        // "encoder": {
        //     "type": "lstm",
        //     "input_size": 300,
        //     "hidden_size": 512,
        //     "num_layers": 2,
        //     "bidirectional": true,
        //     "dropout": 0.5
        // },
        "encoder": {
            "type": "gru",
            "input_size": 300,
            "hidden_size": 512,
            "num_layers": 2,
            "bidirectional": true,
            "dropout": 0.5
            // "positional_encoding": "embedding",
        },
        "max_decoding_steps": 20,
        "beam_size": 1,
        "attention": {
            "type":"additive",
            "vector_dim": 512,
            "matrix_dim": 512,
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
            // "sorting_keys": ["source_tokens"],
        },
    },
    "trainer": {
        "num_epochs": 80,
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-5
        },
        "learning_rate_scheduler": {
            "type": "step",
            "step_size": 20,
            "gamma": 0.5
        },
        // "optimizer": {
        //     "type": "huggingface_adamw",
        //     "lr": 1e-3,
        //     "betas": [0.9, 0.999],
        //     "eps": 1e-8,
        //     "correct_bias": true
        // },
        // "learning_rate_scheduler": {
        //     "type": "polynomial_decay",
        // },
        // "grad_norm": 1.0,
        // "use_amp": true,
    },
    //"random_seed": 42, //42
    //"numpy_seed": 42,
    //"pytorch_seed": 42,
}