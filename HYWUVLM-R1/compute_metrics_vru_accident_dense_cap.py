import json
from collections import defaultdict
import os
import string
import torch
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from pycocoevalcap.spice.spice import Spice
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
nltk.download('punkt')


def evaluate_spice(refs, hyps):
    """
    refs, hyps: two lists of 1000 strings each (reference & hypothesis captions)
    Returns: (overall_f1, individual_f1_list)
    """
    assert len(refs) == len(hyps) == 1000
 
    gts = {str(i): [refs[i]] for i in range(1000)}
    res = {str(i): [hyps[i]] for i in range(1000)}
 
    scorer = Spice()
    overall_f1, scores = scorer.compute_score(gts, res)
 
    # scores is a list of dicts per sample; we extract the F1 under 'All'
    f1_list = [score_dict["All"]["f"] for score_dict in scores]
 
    return overall_f1, f1_list

def evaluate_meteor(references, hypotheses):
    """
    Compute average METEOR score.
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = [word for word in word_tokenize(ref) if word not in string.punctuation]
        hyp_tokens = [word for word in word_tokenize(hyp) if word not in string.punctuation]
        scores.append(meteor_score([ref_tokens], hyp_tokens))
    return sum(scores) / len(scores)
 
def evaluate_rouge(references, hypotheses):
    """
    Compute average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {r: {'precision': 0, 'recall': 0, 'fmeasure': 0} for r in ['rouge1', 'rouge2', 'rougeL']}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for r_type, r_score in scores.items():
            total_scores[r_type]['precision'] += r_score.precision
            total_scores[r_type]['recall'] += r_score.recall
            total_scores[r_type]['fmeasure'] += r_score.fmeasure
 
    average_scores = {
        r: {
            'precision': total_scores[r]['precision'] / len(references),
            'recall': total_scores[r]['recall'] / len(references),
            'fmeasure': total_scores[r]['fmeasure'] / len(references),
        }
        for r in total_scores
    }
    return average_scores


def Dense_Captioning_eval():

    # Load COMET model
    comet_path = download_model("Unbabel/wmt22-comet-da") 
    comet = load_from_checkpoint(comet_path)
    save_path = f"./results/Dense_Captioning/InternVL2-8B_DenseCaption_Score.json"

    all_gen_captions = []
    all_gt_captions = []


    caption_path = "scores_vru_accident_densecap/all.jsonl"
    with open(caption_path, 'r') as infile:
        for line in infile.readlines():
            if not line.strip():
                continue
            data = json.loads(line.strip())
            clean_answer = (
                data['answer']
                .replace("<answer>", "")
                .replace("</answer>", "")
                .strip()
            )
            #breakpoint()
            all_gen_captions.append( clean_answer )
            all_gt_captions.append( data['gt_answer'] )
    

    #all_gen_captions.extend(gen_captions)
    #all_gt_captions.extend(gt_captions)
    assert len(all_gen_captions) == len(all_gt_captions), f" Mismatch in caption counts!"
    
    print("Evaluating SPCIE...")
    avg_spice = evaluate_spice(all_gt_captions, all_gen_captions)

    print("Evaluating METEOR...")
    avg_meteor = evaluate_meteor(all_gt_captions, all_gen_captions)

    print("Evaluating ROUGE...")
    rouge_scores = evaluate_rouge(all_gt_captions, all_gen_captions)
    


    print("Evaluating COMET...")
    comet_data = [
    {"src": "", "mt": all_gen_captions[i], "ref": all_gt_captions[i]}
    for i in range(len(all_gt_captions))
    ]
    model_output = comet.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    comet_scores = model_output.scores
    avg_comet_score = sum(comet_scores) / len(comet_scores)
    
    avg_spice = avg_spice[0]

    result_dict = {
        "SPICE": round(avg_spice, 3),
        "METEOR": round(avg_meteor, 3),
        "COMET": round(avg_comet_score, 3),
        "ROUGE": {
            r_type.upper(): {
                "precision": round(scores["precision"], 3),
                "recall": round(scores["recall"], 3),
                "fmeasure": round(scores["fmeasure"], 3)
            } for r_type, scores in rouge_scores.items()
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"\n Evaluation results saved to {save_path}")

if __name__ == "__main__":
    Dense_Captioning_eval()