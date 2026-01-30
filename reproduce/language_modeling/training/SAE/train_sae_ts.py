import sys
from pathlib import Path

ROOT = Path().resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os

import torch
from codecarbon import track_emissions
from datasets import load_from_disk
from einops import *
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from language.transformer import Transformer
from sae import SAE, Point, SAEConfig

device = "cuda"
name = "Julianvn/facts_ts-new"  # TODO: change with your own trained tinystories model
model = Transformer.from_pretrained(name).to(device)
SAE_CTX = 128

dataset = (
    load_from_disk("tinystories_tokenized_clean")
    .select_columns(["input_ids"])
    .with_format("torch")
)  # TODO: change with your own cleaned tinystories dataset


@track_emissions(output_file="emissions_sae_ts.csv")
def main():
    def train_sae(model, dataset, layer, point_name, tag="paper-repro"):
        print(f"--- Training SAE for Layer {layer} : {point_name} ---")

        config = SAEConfig(
            point=Point(point_name, layer),
            target=None,
            d_model=model.config.d_model,
            n_ctx=SAE_CTX,
            expansion=4,
            k=30,
            lr=1e-4,
            out_batch=4096,
            normalize_decoder=True,
            encoder_bias=False,
            in_batch=32,
            n_batches=256,
            n_buffers=75,
            tag=tag,
        )

        sae = SAE(config).to(device).float()

        def tokenize_fn(examples):
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("Julianvn/facts-fw-med-good")

            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=SAE_CTX,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_fn, batched=True, remove_columns=["text"]
        )
        tokenized_dataset = tokenized_dataset.with_format("torch")

        val_loader = DataLoader(tokenized_dataset, batch_size=32)

        max_ctx = SAE_CTX
        val_batches_list = []
        for i, batch in enumerate(val_loader):
            if i >= 5:
                break
            val_batches_list.append(batch["input_ids"][:, :max_ctx].to(device))

        val_batches = torch.cat(val_batches_list, dim=0)

        sae.fit(
            model=model, train=tokenized_dataset, validate=val_batches, project=None
        )

        return sae

    model.eval()
    points = ["mlp-in", "mlp-out"]

    for layer in tqdm(range(6)):
        print(f"=== Training layer {layer} ===")
        for point in points:
            print(f"=== Training layer {layer} Points {point} ===")
            sae = train_sae(model, dataset, layer, point)
            dir_path = f"sae_models/{layer}-{point}-x4-k30"
            os.makedirs(dir_path, exist_ok=True)
            state_dict = sae.cpu().state_dict()
            save_file(state_dict, f"{dir_path}/model.safetensors")
        print(f"=== Finished layer {layer} ===")


if __name__ == "__main__":
    main()
