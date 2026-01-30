import sys
from pathlib import Path

ROOT = Path().resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codecarbon import track_emissions
from datasets import load_from_disk

from language import Transformer


@track_emissions(output_file="emissions_ts.csv")
def train():
    model = Transformer.from_config(
        tokenizer="ts-4096",
        n_layer=6,
        d_model=2 * 256,
        d_hidden=2 * 4 * 256,
        n_head=8,
        bias=True,
        norm_bias=True,
        attention2=True,
    ).cuda()
    model.summary()

    dataset = load_from_disk(
        "./tinystories_tokenized"
    )  # TODO: change with your own cleaned tinystories dataset
    model.fit(
        dataset["train"],
        project="ts_checkpoints",
        num_train_epochs=5,
        wd=0.1,
        batch_size=64,
        gradient_accumulation_steps=8,
        bf16=True,
    )


if __name__ == "__main__":
    train()
