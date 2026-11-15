import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from grapht5.datasets.data import Dataset
from grapht5.modeling.model import FiDT5
from grapht5.pipeline import evaluation
from grapht5.runtime.slurm import get_device, move_to_device
from grapht5.runtime.util import calc_codebleu_safe, weighted_average

logger = logging.getLogger(__name__)


def evaluate(model: FiDT5, dataset: Dataset, dataloader: DataLoader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    exactmatch, golds, preds = [], [], []

    result_path = None
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / "test_results"
        result_path = write_path / f"{opt.name}.json"

    generated_answers = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Testing "):
            (
                idx,
                _labels,
                _target_mask,
                input_ids,
                attention_mask,
                node_ids,
                node_masks,
                node_roles,
                edges,
            ) = move_to_device(batch, get_device(opt, opt.local_rank))

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids=node_ids,
                node_masks=node_masks,
                node_roles=node_roles,
                edges=edges,
                max_length=opt.answer_maxlength,
                num_beams=opt.beam_size,
                num_return_sequences=opt.num_return_sequences,
            )

            if opt.write_crossattention_scores:
                passage_score, graph_score = model.get_crossattention_scores(
                    move_to_device(attention_mask, get_device(opt, opt.local_rank)),
                    move_to_device(node_masks, get_device(opt, opt.local_rank)),
                )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                candidate_rank = k % opt.num_return_sequences + 1
                example_index = k // opt.num_return_sequences
                example = dataset.data[idx[example_index].item()]
                target = example.target

                score = evaluation.ems(ans, [target])
                exactmatch.append(score)
                golds.append([target])
                preds.append(ans)

                if opt.write_results:
                    output = {
                        "id": example.id,
                        "cwe_id": example.cwe_id,
                        "cve_id": example.cve_id,
                        "original_address": example.original_address,
                        "nodes": len(example.vdg.nodes),
                        "links": len(example.vdg.links),
                        "target": example.target,
                        "generated_answer": ans,
                        "candidate_rank": candidate_rank,
                        "is_exact_match": score,
                    }

                    if example.vul4j:
                        output["vul4j"] = example.vul4j

                    if opt.write_crossattention_scores:
                        output["passage_score"] = passage_score[k].item()
                        output["vdg_score"] = graph_score[k].item()

                    generated_answers.append(output)

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f"Process rank:{opt.global_rank}, {i + 1} / {len(dataloader)}"
                if len(exactmatch) == 0:
                    log += "| no answer to compute scores"
                else:
                    log += f" | average = {100 * np.mean(exactmatch):.3f} EM | "

                if len(preds) > 0:
                    bleu = calc_codebleu_safe(golds, preds, lang=opt.language)
                    log += f"{100 * bleu['codebleu']:.2f} CodeBLEU | {100 * bleu['weighted_ngram_match_score']:.2f} BLEU-4"

                logger.warning(log)

    # final output
    bleu = calc_codebleu_safe(golds, preds, lang=opt.language)

    log = f"Process rank:{opt.global_rank}, total {total} | average = {100 * np.mean(exactmatch):.3f} EM | "
    log += f"{100 * bleu['codebleu']:.2f} CodeBLEU | {100 * bleu['weighted_ngram_match_score']:.2f} BLEU-4"

    logger.warning(log)

    if opt.is_distributed:
        torch.distributed.barrier()

    score, total = weighted_average(np.mean(exactmatch), total, opt)
    codebleu = {k: weighted_average(v, total, opt)[0] for k, v in bleu.items()}

    if result_path is not None:
        if opt.is_distributed:
            answer_stub = (
                [None for _ in range(opt.world_size)] if opt.global_rank == 0 else None
            )
            dist.gather_object(generated_answers, answer_stub, 0)

            if opt.is_main:
                results = [
                    sample for sub_result in answer_stub for sample in sub_result
                ]
                results = sorted(results, key=lambda x: (x["id"], x["candidate_rank"]))

                with result_path.open("w") as handle:
                    json.dump(results, handle, indent=1)
        else:
            with result_path.open("w") as handle:
                json.dump(generated_answers, handle, indent=1)

    return score, codebleu, total
