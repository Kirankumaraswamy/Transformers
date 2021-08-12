import glob
import os
import json
import tensorflow as tf
from tensorboard.plugins.hparams import plugin_data_pb2

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

def read_tf_event_file(tfevent_file:str):
    import glob
    all_tf_files_in_folder = []
    for f in glob.glob(tfevent_file + '/*/*.out.*', recursive=True):
        all_tf_files_in_folder.append(f)
    for f in glob.glob(tfevent_file + '/*.out.*', recursive=True):
        all_tf_files_in_folder.append(f)

    run_config = {}
    parameter_config = {}

    for tfevent_file in all_tf_files_in_folder:
        # Read in tensorflow summary file
        event_acc = EventAccumulator(path=tfevent_file)
        event_acc.Reload()


        # Get Scalars
        for scalar_summary_key in event_acc.Tags()['scalars']:
            _, step_nums, values = zip(*event_acc.Scalars(scalar_summary_key))
            run_config[scalar_summary_key] = values

        if 'Hyperparameters/text_summary' in event_acc.Tags():
            # Get Hyperparameters and others:
            for ten_ev in event_acc.Tensors(tag='Hyperparameters/text_summary'):
                vals_byte = (ten_ev.tensor_proto.string_val[0])
                vals = vals_byte.decode('utf-8')

                key, val = vals.replace(" ", "").split(":")
                assert key not in parameter_config.keys()
                parameter_config[key] = val

        if 'Other Parameters/text_summary' in event_acc.Tags():
            for ten_ev in event_acc.Tensors(tag='Other Parameters/text_summary'):
                vals_byte = (ten_ev.tensor_proto.string_val[0])
                vals = vals_byte.decode('utf-8')
                key, val = vals.replace(" ", "").split(":")
                assert key not in parameter_config.keys()
                parameter_config[key] = val

        md_keys = list(event_acc.summary_metadata)
        for k in md_keys:
            data = event_acc.summary_metadata[k]

            plugin_data = plugin_data_pb2.HParamsPluginData.FromString(data.plugin_data.content)
            if plugin_data.HasField("session_start_info"):
                ks = list(plugin_data.session_start_info.hparams.keys())
                for kss in ks:
                    parameter_config[kss] = plugin_data.session_start_info.hparams[kss].number_value
    return run_config, parameter_config


if __name__ == "__main__":
    #run_config, parameter_config = read_tf_event_file(tfevent_file="../20210707073211.604346/events.out.tfevents.1625635947.gpuv100.nemo.privat.18035.0")
    run_config, parameter_config = read_tf_event_file(tfevent_file="/home/kiran/kiran/dl_lab/project/project_experiments/seeds/encdec_warmup/encdec_warmup_kiran/runs/20210708111543.558214")

    print("Normal Scalar Values")
    for k, v in run_config.items():
        print("key ", k, " len ", len(v), " values: ", v)

    print("\n\n")
    print("PARAMETER CONFIG Values")
    for k, v in parameter_config.items():
        print("key ", k, " values: ", v)