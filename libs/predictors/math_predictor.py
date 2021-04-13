from typing import List, Dict, Any
from overrides import overrides
import json

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from libs.tools.evaluation import evaluate_prefix, replace_pseudo_tokens, evaluate_number


@Predictor.register("math")
class MathPredictor(Predictor):
    """
    """

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        Override this method to output Chinese
        """
        return json.dumps(outputs, ensure_ascii=False) + "\n"

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        model_output = self._model.forward_on_instance(instance)
        metadata = instance["metadata"]

        # May contain many predictions and many answers if the beam search size is larger than one.
        original_tokens = replace_pseudo_tokens(
            model_output["predicted_tokens"], metadata["numbers"])
        try:
            predicted_answer = evaluate_prefix(original_tokens)
        except:
            predicted_answer = "NaN"

        # We dump the target for reference
        # Target includes the equation and token ids like "2 + 5" and (2, 3, 0)"
        target_tokens = metadata['target_tokens']

        # Correct answer and equation
        if target_tokens == original_tokens:
            equation_correct = True
        else:
            equation_correct = False
        if (predicted_answer != "NaN" and abs(evaluate_number(metadata["answer"]) - predicted_answer) < 1e-4):
            answer_correct = True
        else:
            answer_correct = False

        output = {
            "id": metadata["id"],
            "problem": metadata["problem"],
            "equation": metadata["equation"],
            "answer": metadata["answer"],
            "numbers": metadata["numbers"],
            "target": target_tokens,
            "predictions": model_output["predictions"],
            "predicted_tokens": model_output["predicted_tokens"],
            "predicted_answer": predicted_answer,
            "equation_correct": equation_correct,
            "answer_correct": answer_correct,
        }

        return sanitize(output)
