import sys
from pathlib import Path

ROOT = Path().resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codecarbon import track_emissions
from datasets import load_dataset

from language import Transformer


@track_emissions(output_file="emissions_fw_12.csv")
def train():
    model_12 = Transformer.from_config(
        tokenizer="mistral",
        n_layer=12,
        n_ctx=512,
        d_model=768,
        d_hidden=4 * 768,
        n_head=12,
        bias=True,
        norm_bias=True,
        attention2=True,
    ).cuda()
    print(model_12.summary())

    train = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    tokenized = train.map(model_12.tokenize, batched=True)
    model_12.fit(
        tokenized,
        project="fw_12_checkpoints",
        max_steps=10_000,
        batch_size=64,
        gradient_accumulation_steps=8,
        bf16=True,
    )


if __name__ == "__main__":
    train()
