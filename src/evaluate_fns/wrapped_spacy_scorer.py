from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple

from spacy.scorer import PRFScore, Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example

import dacy
from dacy.datasets import dane


class DaCyScorer(Scorer):
    @staticmethod
    def score_spans_plus(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Doc, str], Iterable[Span]] = getattr,
        has_annotation: Optional[Callable[[Doc], bool]] = None,
        labeled: bool = True,
        allow_overlap: bool = False,
        **cfg,
    ) -> Tuple[Dict[str, Any], PRFScore]:
        """Returns PRF scores for labeled spans.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Doc, str], Iterable[Span]]): Defaults to getattr. If
            provided, getter(doc, attr) should return the spans for the
            individual doc.
        has_annotation (Optional[Callable[[Doc], bool]]) should return whether a `Doc`
            has annotation for this `attr`. Docs without annotation are skipped for
            scoring purposes.
        labeled (bool): Whether or not to include label information in
            the evaluation. If set to 'False', two spans will be considered
            equal if their start and end match, irrespective of their label.
        allow_overlap (bool): Whether or not to allow overlapping spans.
            If set to 'False', the alignment will automatically resolve conflicts.
        RETURNS: A dictionary containing the PRF scores under
            the keys attr_p/r/f and the per-type PRF scores under attr_per_type.
            AND the PRFScore object.

        DOCS: https://spacy.io/api/scorer#score_spans
        """
        score = PRFScore()
        score_per_type = dict()
        for example in examples:
            pred_doc = example.predicted
            gold_doc = example.reference
            # Option to handle docs without annotation for this attribute
            if has_annotation is not None and not has_annotation(gold_doc):
                continue
            # Find all labels in gold
            labels = set([k.label_ for k in getter(gold_doc, attr)])
            # If labeled, find all labels in pred
            if has_annotation is None or (
                has_annotation is not None and has_annotation(pred_doc)
            ):
                labels |= set([k.label_ for k in getter(pred_doc, attr)])
            # Set up all labels for per type scoring and prepare gold per type
            gold_per_type: Dict[str, Set] = {label: set() for label in labels}
            for label in labels:
                if label not in score_per_type:
                    score_per_type[label] = PRFScore()
            # Find all predidate labels, for all and per type
            gold_spans = set()
            pred_spans = set()
            for span in getter(gold_doc, attr):
                gold_span: Tuple
                if labeled:
                    gold_span = (span.label_, span.start, span.end - 1)
                else:
                    gold_span = (span.start, span.end - 1)
                gold_spans.add(gold_span)
                gold_per_type[span.label_].add(gold_span)
            pred_per_type: Dict[str, Set] = {label: set() for label in labels}
            if has_annotation is None or (
                has_annotation is not None and has_annotation(pred_doc)
            ):
                for span in example.get_aligned_spans_x2y(
                    getter(pred_doc, attr), allow_overlap
                ):
                    pred_span: Tuple
                    if labeled:
                        pred_span = (span.label_, span.start, span.end - 1)
                    else:
                        pred_span = (span.start, span.end - 1)
                    pred_spans.add(pred_span)
                    pred_per_type[span.label_].add(pred_span)
            # Scores per label
            if labeled:
                for k, v in score_per_type.items():
                    if k in pred_per_type:
                        v.score_set(pred_per_type[k], gold_per_type[k])
            # Score for all labels
            score.score_set(pred_spans, gold_spans)
        # Assemble final result
        final_scores: Dict[str, Any] = {
            f"{attr}_p": None,
            f"{attr}_r": None,
            f"{attr}_f": None,
        }
        if labeled:
            final_scores[f"{attr}_per_type"] = None
        if len(score) > 0:
            final_scores[f"{attr}_p"] = score.precision
            final_scores[f"{attr}_r"] = score.recall
            final_scores[f"{attr}_f"] = score.fscore
            if labeled:
                final_scores[f"{attr}_per_type"] = {
                    k: v.to_dict() for k, v in score_per_type.items()
                }
        return final_scores, score


if __name__ == "__main__":
    nlp = dacy.load("small")
    test = dane(splits="test")

    examples = list(test(nlp))

    # apply model
    for e in examples:
        e.predicted = nlp(e.text)

    # Default scoring pipeline
    scorer = DaCyScorer(nlp)

    scores_dict, score_obj = scorer.score_spans_plus(examples, attr="ents")
    # score_obj.tp
    # score_obj.fp
    # score_obj.fn
    print(f"FP: {score_obj.fp}, TP: {score_obj.tp}, FN: {score_obj.fn}, Precision: {score_obj.precision}, Recall: {score_obj.recall} F1: {score_obj.fscore}")
    print(scores_dict)
