import logging

from torch.optim import Adam

from models.transformer_model import MultiHeadedAttention


def get_optimizer(model, training_config: dict):
    print("SCHEDULERS 2")
    if training_config["num_of_optimizers"] == 1:
        return single_optimizer(model=model, training_config=training_config)
    elif training_config["num_of_optimizers"] == 2:
        if training_config["layer_scheme"] == "enc-dec":
            return double_optimizer(model=model, param_function=encoder_decoder, training_config=training_config)
        elif training_config["layer_scheme"] == "ff-att":
            return double_optimizer(model=model, param_function=ff_att, training_config=training_config)
        else:
            layer_scheme = training_config["layer_scheme"]
            raise ValueError(f"Wrong type of --layer_scheme, error was caused by layer_scheme {layer_scheme}")
    else:
        n_o = training_config["num_of_optimizers"]
        raise ValueError(f"Wrong number of optimizers, error was caused by num_of_optimizers {n_o}")


def single_optimizer(model, training_config: dict):
    logging.info("SETUP NUM OPTIMIZERS: 1 ")

    adam1 = Adam(params=model.parameters(), lr=training_config["optimizer_1_lr"], betas=(training_config["optimizer_1_beta1"], training_config["optimizer_1_beta2"]), eps=training_config["optimizer_1_eps"])
    if training_config["scheduler1"] == "paper":
        logging.info("Optimizer 1 is paper")
        custom_lr_optimizer_1 = CustomLRAdamOptimizer(optimizer=adam1, model_dimension=training_config["model_dims"], num_of_warmup_steps=training_config['num_warmup_steps1'])
    elif training_config["scheduler1"] == "no":
        logging.info("Optimizer 1 is no")
        custom_lr_optimizer_1 = JustAdam(optimizer=adam1)
    else:
        raise ValueError("No correct optimizer found")
    custom_lr_optimizer_2 = None
    return custom_lr_optimizer_1, custom_lr_optimizer_2


def double_optimizer(model, param_function, training_config: dict):
    optimizer_1_params, optimizer_2_params = param_function(model)
    adam1 = Adam(params=optimizer_1_params, lr=training_config["optimizer_1_lr"], betas=(training_config["optimizer_1_beta1"], training_config["optimizer_1_beta2"]), eps=training_config["optimizer_1_eps"])
    if training_config["scheduler1"] == "paper":
        logging.info("Optimizer 1 is paper")
        custom_lr_optimizer_1 = CustomLRAdamOptimizer(optimizer=adam1, model_dimension=training_config["model_dims"], num_of_warmup_steps=training_config['num_warmup_steps1'])
    elif training_config["scheduler1"] == "no":
        logging.info("Optimizer 1 is no")
        custom_lr_optimizer_1 = JustAdam(optimizer=adam1)
    else:
        raise ValueError("No correct optimizer found")

    adam2 = Adam(params=optimizer_2_params, lr=training_config["optimizer_2_lr"], betas=(training_config["optimizer_2_beta1"], training_config["optimizer_2_beta2"]), eps=training_config["optimizer_2_eps"])
    if training_config["scheduler2"] == "paper":
        logging.info("Optimizer 2 is paper")
        custom_lr_optimizer_2 = CustomLRAdamOptimizer(optimizer=adam2, model_dimension=training_config["model_dims"], num_of_warmup_steps=training_config['num_warmup_steps2'])
    elif training_config["scheduler2"] == "no":
        logging.info("Optimizer 2 is no")
        custom_lr_optimizer_2 = JustAdam(optimizer=adam2)
    else:
        raise ValueError("No correct optimizer found")

    del optimizer_1_params  # might bring performance
    del optimizer_2_params  # might bring performance

    return custom_lr_optimizer_1, custom_lr_optimizer_2

def encoder_decoder(model):
    # Training with one or two optimizers
    logging.info("SETUP NUM OPTIMIZERS: 2")

    encoder_params = [param for param in model.src_embedding.parameters()]
    encoder_params.extend([param for param in model.src_pos_embedding.parameters()])
    encoder_params.extend([param for param in model.encoder.parameters()])

    decoder_params = [param for param in model.trg_embedding.parameters()]
    decoder_params.extend([param for param in model.trg_pos_embedding.parameters()])
    decoder_params.extend([param for param in model.decoder.parameters()])
    decoder_params.extend([param for param in model.decoder_generator.parameters()])
    return encoder_params, decoder_params


def add_att(model, att_list):
    #print("Attention: ------------ ", model.__class__.__name__)
    if len(list(model.children())) <= 1:
        for param in model.parameters():
            att_list.append(param)
    else:
        for module in model.children():
            add_att(module, att_list)
    return att_list

def get_modules(model, ff_list, att_list):
    #print(model.__class__.__name__)
    if model.__class__.__name__ == "MultiHeadedAttention":
        add_att(model, att_list)
    else:
        if len(list(model.children())) <= 1:
            for param in model.parameters():
                ff_list.append(param)
            #ff_list.append(model)
        else:
            for module in model.children():
                get_modules(module, ff_list, att_list)
    return ff_list, att_list

def ff_att(model):
    # Training with one or two optimizers
    logging.info("SETUP NUM OPTIMIZERS: 2")
    feedfoward_params, attention_params = get_modules(model, [], [])

    return feedfoward_params, attention_params


class CustomLRAdamOptimizer:
    """ Papers Learning rate schedule"""

    def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps

        self.current_step_number = 0

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate

        self.optimizer.step()  # apply gradients

    # Check out the formula at Page 7, Chapter 5.3 "Optimizer" and playground.py for visualization
    def get_current_learning_rate(self):
        # For readability purpose
        step = self.current_step_number
        warmup = self.num_of_warmup_steps

        return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class JustAdam:
    """ No Learning rate schedule"""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
