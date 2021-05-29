local number_of_branch_map = import '../extra_files/number_of_branches.jsonnet';
{
    "train_data_path": "data/geometry_5fold/fold0_train.json",
    "dataset_reader": {
        "type": "math23k",
        "num_token_type": "NUM",
        "use_original_equation": false,
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
        "type": "S2G-KG",
        "target_namespace": "equation_vocab",
        "number_of_branch_map": number_of_branch_map,
        "child_node_generator":{
            "type": "gru",
            "input_size": 128,
            "hidden_size": 512
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
             "sorting_keys": ["source_tokens"],
        }
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