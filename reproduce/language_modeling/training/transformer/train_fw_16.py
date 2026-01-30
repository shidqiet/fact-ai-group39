from codecarbon import track_emissions
from datasets import load_dataset

from language import Transformer


@track_emissions(output_file="emissions_fw_16.csv")
def train():
    model_16 = Transformer.from_config(
        tokenizer="mistral",
        n_layer=16,
        n_ctx=512,
        d_model=1024,
        d_hidden=4 * 1024,
        n_head=16,
        bias=True,
        norm_bias=True,
        attention2=True,
    ).cuda()
    print(model_16.summary())

    train = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    tokenized = train.map(model_16.tokenize, batched=True)
    model_16.fit(
        tokenized,
        project="fw_16_checkpoints",
        max_steps=10_000,
        batch_size=32,
        gradient_accumulation_steps=16,
        bf16=True,
    )


if __name__ == "__main__":
    train()
