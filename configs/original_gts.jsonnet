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
        // "max_instances": 1000 // DEBUG setting
    },
    "vocabulary":{
       "min_count": {
           "tokens": 5,
       },
       "pretrained_files": {
           "target_vocab": "extra_files/equation_vocab.txt",
       },
       "only_include_pretrained_words": true,
    },
    "model": {
        "type": "seq2tree",
        "target_namespace": "target_vocab",
        "number_of_branch_map": number_of_branches,
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    // "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                    "trainable": true
                }
            }
        },
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
        "type": "multiprocess",
        "batch_size": 64,
        "drop_last": false,
        "shuffle": true,
      //  "batch_sampler": {
       //     "type": "bucket",
      //      "batch_size": 64,
            // "sorting_keys": ["source_tokens"],
      //  },
    },
    "trainer": {
        "num_epochs": 80,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-5
        },
        "learning_rate_scheduler": {
            "type": "step",
            "step_size": 20,
            "gamma": 0.5
        },
    },
    "random_seed": null, 
    "numpy_seed": null,
    "pytorch_seed": null,
}