import augmenty 
import spacy
import dacy
import pathlib
from evaluate_fns.wrapped_spacy_scorer import DaCyScorer
import pandas as pd 

def filter_ents(doc, ents_to_keep):
  ents = [e for e in doc.ents if e.label_ in ents_to_keep]
  doc.ents = ents
  return doc

def eval_fairness_metrics(model_dict, augmenters, dataset):
    # define output path
    outfolder = "results_DSH"
    pathlib.Path(outfolder).mkdir(parents=True, exist_ok=True)

    for mdl in model_dict:
        print(f"[INFO]: Scoring model '{mdl}' using DaCy")

        # load model depending on model name (different pipelines)
        if "dacy" in mdl:
            apply_fn = dacy.load(model_dict[mdl])
        elif "spacy" in mdl:
            apply_fn = spacy.load(model_dict[mdl])
            spacy.prefer_gpu()
        else:
            apply_fn = model_dict[mdl]
        
        i = 0
        scores = []
        for aug, nam, k in augmenters:
            print(f"\t Running augmenter: {nam} | Amount of times: {k}")

            # augment
            i += 1
            for n in range(k):    
                augmented_corpus = [e for example in dataset(apply_fn) for e in aug(apply_fn, example)]
            
                for e in augmented_corpus:
                    e.predicted = apply_fn(e.text)

                # initialize scorer
                scorer = DaCyScorer(apply_fn)

                # get scores_dict 
                scores_dict, score_obj = scorer.score_spans_plus(augmented_corpus, attr="ents")
                
                # define values for dataframe
                score_vals = {
                    "model": mdl, 
                    "augmenter":nam, 
                    "i":i-1, 
                    "k":n,
                    "FP":score_obj.fp, 
                    "TP":score_obj.tp, 
                    "FN":score_obj.fn,
                    "Precision":score_obj.precision,
                    "Recall":score_obj.recall,
                    "F1_score":score_obj.fscore
                    }

                # create pandas dataframe
                scores_data = pd.DataFrame.from_records(score_vals, index=[0])

                # reorder columns 
                scores_data = scores_data[["FP", "TP", "FN", "Precision", "Recall", "F1_score", "k", "model", "augmenter", "i"]]

                print(f"FP: {score_obj.fp}, TP: {score_obj.tp}, FN: {score_obj.fn}")

                scores.append(scores_data)

        scores = pd.concat(scores)
        scores.to_csv(f"{outfolder}/{mdl}_fairness_metrics.csv")





if __name__ == "__main__":
    # import data set 
    testdata = dane(splits=["test"], redownload=True, open_unverified_connected=True)

    # define augmenters: augmenter, name, n repetitions 
    n = 20
    augmenters = [
        (dk_aug, "Danish names", n),
        (muslim_aug, "Muslim names", n),
        (f_aug, "Female names", n),
        (m_aug, "Male names", n),
        (muslim_f_aug, "Muslim female names", n),
        (muslim_m_aug, "Muslim male names", n),
        (unisex_aug, "Unisex names", n),
    ]

    # define models to run 
    model_dict = {
        "spacy_small": "da_core_news_sm",
        "spacy_medium": "da_core_news_md",
        "spacy_large": "da_core_news_lg",
    }

    # evaluate 
    eval_fairness_metrics(model_dict, augmenters, testdata)