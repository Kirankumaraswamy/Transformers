import enum

import spacy
import torch
from torchtext import datasets
from torchtext.data import BucketIterator, Field

from utils_train.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

global longest_src_sentence, longest_trg_sentence


class LanguageDirection(enum.Enum):
    E2G = 0,
    G2E = 1


def get_datasets_and_vocabs(dataset_path, language_direction):
    german_to_english = language_direction == LanguageDirection.G2E.name
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    src_tokenizer = tokenize_de if german_to_english else tokenize_en
    trg_tokenizer = tokenize_en if german_to_english else tokenize_de
    src_field_processor = Field(tokenize=src_tokenizer, pad_token=PAD_TOKEN, batch_first=True)
    trg_field_processor = Field(tokenize=trg_tokenizer, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)

    fields = [('src', src_field_processor), ('trg', trg_field_processor)]
    MAX_LEN = 100  # filter out examples that have more than MAX_LEN tokens
    filter_pred = lambda x: len(x.src) <= MAX_LEN and len(x.trg) <= MAX_LEN

    src_ext = '.de' if german_to_english else '.en'
    trg_ext = '.en' if german_to_english else '.de'

    train_dataset, val_dataset, test_dataset = datasets.IWSLT.splits(exts=(src_ext, trg_ext), fields=fields, root=dataset_path, filter_pred=filter_pred)

    MIN_FREQ = 2
    src_field_processor.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
    trg_field_processor.build_vocab(train_dataset.trg, min_freq=MIN_FREQ)

    return train_dataset, val_dataset, src_field_processor, trg_field_processor


def batch_size_fn(new_example, count, sofar):  # sofar is needed otherwise Code breaks
    """ sets the batch length to number of tokens """
    global longest_src_sentence, longest_trg_sentence

    if count == 1:
        longest_src_sentence = 0
        longest_trg_sentence = 0

    longest_src_sentence = max(longest_src_sentence, len(new_example.src))
    longest_trg_sentence = max(longest_trg_sentence, len(new_example.trg) + 2)
    num_of_tokens_in_src_tensor = count * longest_src_sentence
    num_of_tokens_in_trg_tensor = count * longest_trg_sentence
    return max(num_of_tokens_in_src_tensor, num_of_tokens_in_trg_tensor)


def get_data_loaders(dataset_path, language_direction, batch_size, device):
    train_dataset, val_dataset, src_field_processor, trg_field_processor = get_datasets_and_vocabs(dataset_path, language_direction)

    train_token_ids_loader, val_token_ids_loader = BucketIterator.splits(datasets=(train_dataset, val_dataset),
                                                                         batch_size=batch_size,
                                                                         device=device,
                                                                         sort_within_batch=True,  # this part is really important otherwise we won't group similar length sentences
                                                                         batch_size_fn=batch_size_fn)
    return train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor


def get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id):
    batch_size = src_token_ids_batch.shape[0]
    src_mask = (src_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # src_mask shape = (B, 1, 1, S)
    num_src_tokens = torch.sum(src_mask.long())
    return src_mask, num_src_tokens


def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # shape = (B, 1, 1, T)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)
    num_trg_tokens = torch.sum(trg_padding_mask.long())
    return trg_mask, num_trg_tokens


def get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device):
    src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    trg_mask, num_trg_tokens = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
    return src_mask, trg_mask, num_src_tokens, num_trg_tokens


def get_src_and_trg_batches(token_ids_batch):
    src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg
    trg_token_ids_batch_input = trg_token_ids_batch[:, :-1]
    trg_token_ids_batch_gt = trg_token_ids_batch[:, 1:].reshape(-1, 1)
    return src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt
