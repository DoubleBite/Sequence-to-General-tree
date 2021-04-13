CONFIG="configs/seq2tree.jsonnet"
PACKAGE="libs"
DATA_DIR="data/geometry_5fold_raw"
ROOT_DIR="results/seq2tree"

### Settings
EXP_NAME=$1
SETTINGS=""
ROOT_DIR="${ROOT_DIR}/${EXP_NAME}"

### Cross-validation
for i in {0..4}; do

    RESULT_DIR="${ROOT_DIR}/fold${i}"
    TRAIN_PATH="${DATA_DIR}/fold${i}_train.json"
    # Just use the test set as dev to get the best performance for baslines.
    TEST_PATH="${DATA_DIR}/fold${i}_test.json"

    python -m allennlp train \
        $CONFIG \
        --serialization-dir $RESULT_DIR \
        --include-package $PACKAGE \
        -o "{'train_data_path':'${TRAIN_PATH}', \
             ${SETTINGS}}" \
        -f
      #       'validation_data_path':'${TEST_PATH}',\

    python -m allennlp predict \
        ${RESULT_DIR}/model.tar.gz \
        $TEST_PATH \
        --include-package $PACKAGE \
        --output-file ${RESULT_DIR}/predictions.jsonl \
        --use-dataset-reader \
        --predictor math
done
