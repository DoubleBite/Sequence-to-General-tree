local equation_vocab_str = importstr '../extra_files/equation_vocab.txt';
local number_of_branches = import '../extra_files/number_of_branches.jsonnet';
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
        "target_namespace": "target_vocab",
        "number_of_branch_map": number_of_branches,
        "max_decoding_steps": 20,
        "beam_size": 1,
        "child_generator":{
            "type": "gru",
            "input_size": 128,
            "hidden_size": 512
        }
    },
    "data_loader": {
        "batch_size": 64,
        "drop_last": false,
        "shuffle": true,
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