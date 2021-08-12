import argparse
import json
import logging
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.LabelSmoothing import LabelSmoothingDistribution
from models.transformer_model import Transformer
from utils_data.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, LanguageDirection
from utils_development.summary import *
from utils_train.all_configs import get_training_config
from utils_train.bleu_score import calculate_bleu_score
from utils_train.constants import *
from utils_train.model_saving_utils import get_training_state
from utils_train.schedulers import get_optimizer
import random
import numpy as np

num_of_trg_tokens_processed = 0
bleu_scores = {}
global_train_step, global_val_step = [0, 0]
writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
logging.basicConfig(filename=f'{LOG_FILE}.log', level=logging.DEBUG)


def get_train_val_loop(transformer_model, custom_lr_optimizer_1, custom_lr_optimizer_2, kl_div_loss, label_smoothing, pad_token_id, time_start):
    def train_val_loop(is_train, token_ids_loader, epoch):
        global num_of_trg_tokens_processed, global_train_step, global_val_step, writer

        if is_train:
            transformer_model.train()
        else:
            transformer_model.eval()

        device = next(transformer_model.parameters()).device

        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            if DEBUG_CODE and batch_idx == DEBUG_CODE_TRAIN_EPOCH_LENGTH: logging.info("DEBUG_CODE BREAK"); break  # TODO: delete later

            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)

            predicted_log_distributions = transformer_model(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)  # KL loss expects log probabilities
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

            if is_train and training_config["num_of_optimizers"] == 1:
                custom_lr_optimizer_1.zero_grad()
            elif is_train and training_config["num_of_optimizers"] == 2:
                custom_lr_optimizer_1.zero_grad()
                custom_lr_optimizer_2.zero_grad()

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train and training_config["num_of_optimizers"] == 1:
                loss.backward()
                custom_lr_optimizer_1.step()
            elif is_train and training_config["num_of_optimizers"] == 2:
                loss.backward()
                custom_lr_optimizer_1.step()
                custom_lr_optimizer_2.step()
            else:
                pass  # this is validation

            # Logging and metrics
            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens
                writer.add_scalar('training_loss', loss.item(), global_train_step)
                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    log_string = f'Transformer training: time elapsed= {(time.time() - time_start):.2f} [s] ' + f'| epoch={epoch + 1} | batch= {batch_idx + 1} ' + f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]}'
                    logging.info(log_string)
                    num_of_trg_tokens_processed = 0
            else:
                global_val_step += 1
                writer.add_scalar('val_loss', loss.item(), global_val_step)

    return train_val_loop


def train_transformer(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"torch.device: {device}")

    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(dataset_path=training_config['dataset_path'],
                                                                                                              language_direction=training_config['language_direction'],
                                                                                                              batch_size=training_config['batch_size'],
                                                                                                              device=device)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    transformer = Transformer(model_dimension=training_config["model_dims"],
                              src_vocab_size=src_vocab_size,
                              trg_vocab_size=trg_vocab_size,
                              number_of_heads=training_config["num_of_heads"],
                              number_of_layers=training_config["num_of_layers"],
                              dropout_probability=training_config["dropout_prob"]
                              ).to(device)

    # pt_names_and_layers_MultiHeadedAttention_print(model=transformer)
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    label_smoothing = LabelSmoothingDistribution(training_config["smoothing_value"], pad_token_id, trg_vocab_size, device)

    custom_lr_optimizer1, custom_lr_optimizer_2 = get_optimizer(model=transformer, training_config=training_config)

    train_val_loop = get_train_val_loop(transformer_model=transformer,
                                        custom_lr_optimizer_1=custom_lr_optimizer1,
                                        custom_lr_optimizer_2=custom_lr_optimizer_2,
                                        kl_div_loss=kl_div_loss,
                                        label_smoothing=label_smoothing,
                                        pad_token_id=pad_token_id,
                                        time_start=time.time())

    for epoch in range(training_config['num_of_epochs']):
        # Training
        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)

        if (training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0) and not (epoch + 1) == training_config["num_of_epochs"]:
            # save when checkpoint_freq is met, except the last epoch will be saved in the else part below
            ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
            torch.save(get_training_state(training_config, transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            logging.info(f"IF PART:  {ckpt_model_name}")

        if (epoch + 1) == training_config["num_of_epochs"]:
            ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
            torch.save(get_training_state(training_config, transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            logging.info(f"LAST SAVE PART:  {ckpt_model_name}")

        # Validation
        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)

            bleu_score = calculate_bleu_score(transformer, val_token_ids_loader, trg_field_processor)
            writer.add_scalar('bleu_score', bleu_score, epoch)
            bleu_scores[f"bleu_score_{epoch + 1}"] = bleu_score


def hyperparameter_tensorboard_log(summary_writer: torch.utils.tensorboard.SummaryWriter, hyperparameters: list, training_config: dict, metrics: dict):
    hyperparameter_dict = dict()
    for hp in hyperparameters:
        hyperparameter_dict[hp] = training_config[hp]

    summary_writer.add_hparams(hparam_dict=hyperparameter_dict, metric_dict=metrics)


if __name__ == "__main__":
    hyperparameters = []  # to write the args into tensorboard hyperparameter log tab
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_optimizers", type=int, choices=[1, 2], help="number of optimizers", default=2)
    hyperparameters.append("num_of_optimizers")
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    hyperparameters.append("num_of_epochs")
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=500)
    hyperparameters.append("batch_size")
    parser.add_argument("--num_of_layers", type=int, help="number of overall layers", default=6)
    hyperparameters.append("num_of_layers")
    parser.add_argument("--model_dims", type=int, help="number of neurons", default=512)
    hyperparameters.append("model_dims")
    parser.add_argument("--num_of_heads", type=int, help="number of heads for attention", default=8)
    hyperparameters.append("num_of_heads")
    parser.add_argument("--dropout_prob", type=float, help="probability for dropout", default=0.0)
    hyperparameters.append("dropout_prob")
    parser.add_argument("--smoothing_value", type=float, help="smoothing_value", default=0.1)
    hyperparameters.append("smoothing_value")

    parser.add_argument("--scheduler1", type=str, choices=["paper", "no"], help="scheduler 1 encoder or complete model if num opt = 1", default="paper")
    hyperparameters.append("scheduler1")
    parser.add_argument("--scheduler2", type=str, choices=["paper", "no"], help="scheduler 2 decoder ", default="paper")
    hyperparameters.append("scheduler2")

    parser.add_argument("--optimizer_1_lr", type=float, help="learning rate optimizer 1", default=0.1)
    hyperparameters.append("optimizer_1_lr")
    parser.add_argument("--num_warmup_steps1", type=int, help="number of warmup steps", default=4000)
    hyperparameters.append("num_warmup_steps1")
    parser.add_argument("--optimizer_1_beta1", type=float, help="learning rate optimizer 1", default=0.9)
    hyperparameters.append("optimizer_1_beta1")
    parser.add_argument("--optimizer_1_beta2", type=float, help="learning rate optimizer 1", default=0.98)
    hyperparameters.append("optimizer_1_beta2")
    parser.add_argument("--optimizer_1_eps", type=float, help="learning rate optimizer 1", default=1e-9)
    hyperparameters.append("optimizer_1_eps")
    parser.add_argument("--optimizer_2_lr", type=float, help="learning rate optimizer 1", default=0.1)
    hyperparameters.append("optimizer_2_lr")
    parser.add_argument("--num_warmup_steps2", type=int, help="number of warmup steps", default=4000)
    hyperparameters.append("num_warmup_steps2")
    parser.add_argument("--optimizer_2_beta1", type=float, help="learning rate optimizer 1", default=0.98)
    hyperparameters.append("optimizer_2_beta1")
    parser.add_argument("--optimizer_2_beta2", type=float, help="learning rate optimizer 1", default=0.88)
    hyperparameters.append("optimizer_2_beta2")
    parser.add_argument("--optimizer_2_eps", type=float, help="learning rate optimizer 1", default=1e-9)
    hyperparameters.append("optimizer_2_eps")

    # Data related args
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    hyperparameters.append("language_direction")

    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=5)

    parser.add_argument("--use_config_load", type=str, choices=["True", "False"], help="load config from json-file", default="False")
    parser.add_argument("--array_number", type=int, help="the array number (optional) only relevant when --use_config_load True", default=0)

    parser.add_argument("--layer_scheme", type=str, choices=["enc-dec", "ff-att"], help="optimize by layer type", default="ff-att")
    hyperparameters.append("layer_scheme")

    parser.add_argument("--name", type=str, choices=["andy", "dakhila", "kiran"], help="your name for loading the experiments and as random seed")

    args = parser.parse_args()


    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    YOURNAME = training_config["name"]

    # For batch jobs overwrite the config with the one saved in utils_train/all_configs.py
    if args.use_config_load == "True":
        array_number = args.array_number -1

        loaded_config = get_training_config(name=YOURNAME)
        loaded_config = loaded_config[array_number]
        for key, value in loaded_config.items():
            training_config[key] = value

    # RANDOM SEEDS
    name_as_int = int.from_bytes(YOURNAME.encode(), "little") % 10000
    r_seed = int(name_as_int * training_config["array_number"])
    random.seed(r_seed)
    torch.manual_seed(r_seed)
    np.random.seed(r_seed)

    # Save parameters, hyperparameters etc to json
    args_json = json.dumps(training_config)
    filename = f'{JSON_FILE_DIR}/parameters.json'
    with open(filename, "w") as outfile:
        outfile.write(args_json)

    # Save all parameters to Tensorboard as Text
    for k, v in training_config.items():
        if k not in hyperparameters:
            writer.add_text(tag="Other Parameters", text_string=f"{k} : {v}")
        else:
            writer.add_text(tag="Hyperparameters", text_string=f"{k} : {v}")

    for k, v in training_config.items():
        logging.info(f"training_config[{k}]:{v}")

    logging.info("\nSTARTING TRAIN CALL")
    train_transformer(training_config)

    logging.info("\nWRITING HYPERPARAMETERS TO TENSORBOARDs hyperparameter selector ")
    hyperparameter_tensorboard_log(summary_writer=writer, hyperparameters=hyperparameters, training_config=training_config, metrics=bleu_scores)
