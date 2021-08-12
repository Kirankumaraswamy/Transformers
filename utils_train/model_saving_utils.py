def get_training_state(training_config, model):
    training_state = {"language_direction": training_config['language_direction'],
                      "num_of_epochs": training_config['num_of_epochs'],
                      "batch_size": training_config['batch_size'],
                      "state_dict": model.state_dict()}
    return training_state
