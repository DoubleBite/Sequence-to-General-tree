from typing import List, Dict, Any
from overrides import overrides
import json

from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict, sanitize

from libs.tools.evaluation import evaluate_prefix, replace_pseudo_tokens, evaluate_number


@Predictor.register("math")
class MathPredictor(Predictor):
    """
    """

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        Override this method to output Chinese.
        """
        return json.dumps(outputs, ensure_ascii=False) + "\n"

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:

        metadata = instance["metadata"]
        model_output = self._model.forward_on_instance(instance)

        # Replace the pseudo tokens with the numbers (e.g., <N0> to 5 )
        predicted_equation = replace_pseudo_tokens(
            model_output["predicted_tokens"], metadata["numbers"])

        # Calculated the predicted answer
        try:
            predicted_answer = evaluate_prefix(predicted_equation)
        except:
            predicted_answer = "NaN"

        # Check whether the equation and the answer is correct
        if metadata['target_tokens'] == model_output["predicted_tokens"]:
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
            "target": metadata['target_tokens'],
            "predicted_tokens": model_output["predicted_tokens"],
            "predicted_answer": predicted_answer,
            "equation_correct": equation_correct,
            "answer_correct": answer_correct,
        }

        return sanitize(output)
