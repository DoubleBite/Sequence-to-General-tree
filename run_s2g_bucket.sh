PACKAGE="libs"
CONFIG="configs/s2g_bucket.jsonnet"
DATASET="data/geometry_5fold_0503"

# It will have five sub-directories for the results of 5-fold cross validation
RESULT_DIR="results/${1}"

# 5-fold cross-validation
for i in {0..4}; do

    TRAIN_PATH="${DATASET}/fold${i}_train.json"
    TEST_PATH="${DATASET}/fold${i}_test.json"
    SUB_DIR="${RESULT_DIR}/fold${i}"

    # Training
    python -m allennlp train \
        $CONFIG \
        --serialization-dir $SUB_DIR \
        --include-package $PACKAGE \
        -o "{'train_data_path':'${TRAIN_PATH}'}" \
        -f

    # Inference
    python -m allennlp predict \
        ${SUB_DIR}/model.tar.gz \
        $TEST_PATH \
        --include-package $PACKAGE \
        --output-file ${SUB_DIR}/predictions.jsonl \
        --use-dataset-reader \
        --predictor math
done
