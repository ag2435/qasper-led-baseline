"""
Albert: I don't think we need this file since we're using the Hugging Face Transformers
version of the model.

See predict() method in qasper/models/qasper.py instead.
"""

import json
from overrides import overrides

from allennlp.common import JsonDict
from allennlp.predictors import Predictor

@Predictor.register('qasper')
class QasperPredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps({"question_id": outputs["question_id"],
                           "predicted_answer": outputs["predicted_answers"],
                           "predicted_evidence": outputs["predicted_evidence"]}) + "\n"
