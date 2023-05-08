from evaluate_fns.augmentation import dk_aug, muslim_aug, f_aug, m_aug, muslim_f_aug, muslim_m_aug, unisex_aug
import dacy
from dacy.datasets import dane

from evaluate_fns.wrapped_spacy_scorer import DaCyScorer 

import spacy

def main(): 
    # define model 
    nlp = dacy.load("large")
    testset = dane(splits="test")

    examples = list(testset(nlp))

    augmented_corpus = [e for example in examples for e in muslim_aug(nlp, example)] 

    scorer = DaCyScorer(nlp)
         
    for e in augmented_corpus:
        e.predicted = nlp(e.text) # e.text or e.predicted ? 
    
    scores_dict, score_obj = scorer.score_spans_plus(examples, attr="ents")

    print(f"FP: {score_obj.fp}, TP: {score_obj.tp}, FN: {score_obj.fn}, Precision: {score_obj.precision}, Recall: {score_obj.recall} F1: {score_obj.fscore}")
    print(scores_dict)

if __name__ == "__main__":
    main()