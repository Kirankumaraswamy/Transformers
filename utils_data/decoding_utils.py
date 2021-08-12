import torch

from utils_data.data_utils import get_masks_and_count_tokens_trg
from utils_train.constants import *


def greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=100):
    device = next(baseline_transformer.parameters()).device
    pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    target_sentences_tokens = [[BOS_TOKEN] for _ in range(src_representations_batch.shape[0])]
    trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[tokens[0]]] for tokens in target_sentences_tokens], device=device)

    # Set to true for a particular target sentence once it reaches the EOS (end-of-sentence) token
    is_decoded = [False] * src_representations_batch.shape[0]

    while True:
        trg_mask, _ = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)  # Shape = (B*T, V)
        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens - 1::num_of_trg_tokens]
        most_probable_last_token_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()
        predicted_words = [trg_field_processor.vocab.itos[index] for index in most_probable_last_token_indices]

        for idx, predicted_word in enumerate(predicted_words):
            target_sentences_tokens[idx].append(predicted_word)

            if predicted_word == EOS_TOKEN:
                is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens == max_target_tokens:
            break

        trg_token_ids_batch = torch.cat((trg_token_ids_batch, torch.unsqueeze(torch.tensor(most_probable_last_token_indices, device=device), 1)), 1)

    target_sentences_tokens_post = []
    for target_sentence_tokens in target_sentences_tokens:
        try:
            target_index = target_sentence_tokens.index(EOS_TOKEN) + 1
        except:
            target_index = None

        target_sentence_tokens = target_sentence_tokens[:target_index]
        target_sentences_tokens_post.append(target_sentence_tokens)

    return target_sentences_tokens_post
