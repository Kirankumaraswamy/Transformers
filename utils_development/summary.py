import torch
from models.transformer_model import MultiHeadedAttention


def pt_names_print(model: torch.nn.Module):
    print("#" * 15 + "     NAME      " + "#" * 15)
    print("# Generated with:")
    print("#     for name, layer in transformer.named_modules():")
    print("#             print(name)")
    print("\n")
    for name, layer in model.named_modules():
        print(name)
    print("\n")
    print("\n")


def pt_names_and_layers_print(model: torch.nn.Module):
    print("#" * 15 + "     NAME,LAYER      " + "#" * 15)
    print("# Generated with:")
    print("#     for name, layer in transformer.named_modules():")
    print("#             print(name, layer)")
    print("\n")
    for name, layer in model.named_modules():
        print(name, layer)
    print("\n")
    print("\n")


def pt_model_print(model: torch.nn.Module):
    print("#" * 15 + "     MODEL      " + "#" * 15)
    print("# Generated with:")
    print("#     print(transformer)")
    print("\n")
    print(model)


def pytorch_model_summary_print(model: torch.nn.Module):
    pt_names_print(model=model)
    pt_names_and_layers_print(model=model)
    pt_model_print(model=model)


def pt_names_and_layers_MultiHeadedAttention_print(model: torch.nn.Module):
    print("#" * 15 + "     NAME,LAYER      " + "#" * 15)
    print("# Generated with:")
    print("#     for name, layer in transformer.named_modules():")
    print("#             print(name, layer)")
    print("\n")
    for name, layer in model.named_modules():
        if isinstance(layer, MultiHeadedAttention):
            print(name, layer)
    print("\n")
    print("\n")
