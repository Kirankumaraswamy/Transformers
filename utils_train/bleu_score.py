import logging
import time

import torch
from nltk.translate.bleu_score import corpus_bleu

from utils_data.data_utils import get_masks_and_count_tokens_src
from utils_data.decoding_utils import greedy_decoding
from utils_train.constants import PAD_TOKEN


def calculate_bleu_score(transformer, token_ids_loader, trg_field_processor):
    """ Calculates the BLEU-4 score. returns score between 0.0 and 1.0"""
    with torch.no_grad():
        pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

        gt_sentences_corpus = []
        predicted_sentences_corpus = []

        ts = time.time()
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg
            if batch_idx % 10 == 0:
                logging.info(f'calculate_bleu_score: batch={batch_idx}, time elapsed = {time.time() - ts} seconds.')

            src_mask, _ = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
            src_representations_batch = transformer.encode(src_token_ids_batch, src_mask)

            predicted_sentences = greedy_decoding(transformer, src_representations_batch, src_mask, trg_field_processor)
            predicted_sentences_corpus.extend(predicted_sentences)

            trg_token_ids_batch = trg_token_ids_batch.cpu().numpy()
            for target_sentence_ids in trg_token_ids_batch:
                target_sentence_tokens = [trg_field_processor.vocab.itos[id] for id in target_sentence_ids if id != pad_token_id]
                gt_sentences_corpus.append([target_sentence_tokens])  # add them to the corpus of GT translations

        bleu_score = corpus_bleu(gt_sentences_corpus, predicted_sentences_corpus)  # * 100.0  # times 100 because corpus_bleu returns between 0.0 and 1.0 with 1.0 equals perfect
        logging.info(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(gt_sentences_corpus)}, time elapsed = {time.time() - ts} seconds.')
        return bleu_score
