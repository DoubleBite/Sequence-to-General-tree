PACKAGE="libs"
CONFIG="configs/s2gkg_bucket.jsonnet"
DATASET="data/geometryQA_5fold"

# It will create a "main directory" in "results".
# The main directory will have five "sub-directories" for 5-fold cross validation.
FIVE_FOLD_DIR="results/${1}"

# 5-fold cross-validation
for i in {0..4}; do

    TRAIN_PATH="${DATASET}/fold${i}_train.json"
    TEST_PATH="${DATASET}/fold${i}_test.json"
    RESULT_DIR="${FIVE_FOLD_DIR}/fold${i}"

    # Training
    python -m allennlp train \
        $CONFIG \
        --serialization-dir $RESULT_DIR \
        --include-package $PACKAGE \
        -o "{'train_data_path':'${TRAIN_PATH}'}" \
        -f

    # Inference
    python -m allennlp predict \
        ${RESULT_DIR}/model.tar.gz \
        $TEST_PATH \
        --include-package $PACKAGE \
        --output-file ${RESULT_DIR}/predictions.jsonl \
        --use-dataset-reader \
        --predictor math \
        --cuda-device 0
done
