import sys
from pathlib import Path

ROOT = Path().resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from einops import *
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from language import Sight
from sae.sae import SAE, Point
from sae.samplers import MultiSampler


class Visualizerfw:
    def __init__(self, model, sae, dataset=None):
        path = hf_hub_download(
            repo_id="Julianvn/facts-fw-small-scope",
            filename="fw-top-activations.safetensors",
        )
        with safe_open(path, framework="pt", device="cpu") as f:
            self.top_activations = f.get_tensor(f"{sae.point.layer}-{sae.point.name}")

        self.sae = sae
        self.sight = Sight(model)

        self.n_ctx = model.config.n_ctx

        self.dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format(
            "torch"
        )

    @staticmethod
    def create_dataset_subset(model):
        tokenized = model.dataset().map(model.tokenize, batched=True)
        input_ids = torch.stack([batch["input_ids"] for batch in tokenized.take(2**14)])

        dataset = Dataset.from_dict({"input_ids": input_ids})
        dataset.push_to_hub(f"tdooms/{model.config.dataset}-16k")

        return dataset

    @staticmethod
    def compute_max_activations(model, dataset=None, in_batch=16):
        dataset = (
            Visualizerfw.create_dataset_subset(model) if dataset is None else dataset
        )

        device = "cpu"

        points = [
            Point(name, layer)
            for layer in range(model.config.n_layer)
            for name in ["mlp-in", "mlp-out"]
        ]
        saes = [
            SAE.from_pretrained(
                "Julianvn/facts-fw-med-new-scope", point=point, expansion=4, k=30
            ).cpu()
            for point in points
        ]

        sampler = MultiSampler(
            Sight(model),
            points,
            dataset=dataset,
            d_model=model.config.d_model,
            n_ctx=128,
            in_batch=in_batch,
        )

        total = 2**14 // in_batch
        tops = []

        for batch, _ in tqdm(zip(sampler, range(total)), total=total):
            top = [
                saes[i].encode(batch["activations"][:, i]).max(1).values
                for i, _ in enumerate(saes)
            ]
            tops.append(top)

        transposed = list(zip(*tops))
        stacked = [torch.cat(t, dim=0) for t in transposed]
        idxs = [s.topk(dim=0, k=100).indices for s in stacked]
        tensors = {f"{p.layer}-{p.name}": idx for idx, p in zip(idxs, points)}

        save_file(tensors, "fw-top-activations.safetensors")

    @staticmethod
    def color_str(str, color, value):
        r, g, b = color
        pre = " " if str.startswith("▁") else ""
        return (
            pre + str[len(pre) :]
            if value == 0
            else f"{pre}\033[48;2;{int(r)};{int(g)};{int(b)}m{str[len(pre) :]}\033[0m"
        )

    @staticmethod
    def color_line(line, colors, values, view):
        idx = values.argmax(dim=-1)
        start, end = max(0, idx + view.start), min(len(line), idx + view.stop)
        return "".join(
            [
                Visualizerfw.color_str(line[i], colors[i], values[i])
                for i in range(start, end)
            ]
        )

    def color_input_ids(self, input_ids, feature=0, view=range(-10, 20), dark=False):
        input_ids = input_ids[:, : self.n_ctx]

        with torch.no_grad(), self.sight.trace(input_ids, validate=False, scan=False):
            features = self.sae.encode(self.sight[self.sae.point]).save()

        values = features[..., feature]
        tokens = [self.sight.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        maxes = values.max(dim=-1, keepdim=True).values
        denom = maxes.where(maxes > 0, torch.ones_like(maxes))
        normalized = (values / denom) * 0.6

        colors = (
            plt.cm.magma(normalized.cpu())[..., :3]
            if dark
            else plt.cm.Blues(normalized.cpu())[..., :3]
        )
        colors = (colors * 255).astype(int)

        for line, color, value in zip(tokens, colors, values):
            print(
                f"{value.max().item():<4.1f}:  {Visualizerfw.color_line(line, color, value, view)}"
            )

    def show_logit_influence(self, feature):
        sims = einsum(self.sae.w_dec.weight[:, feature], self.sight.w_u, "d, b d -> b")
        pos, neg = sims.topk(k=5), sims.topk(k=5, largest=False)

        for idx, val in zip(pos.indices, pos.values):
            print(f"{self.sight.tokenizer.decode(idx)}: {val.item():.2f}", end=", ")

        for idx, val in zip(neg.indices, neg.values):
            print(f"{self.sight.tokenizer.decode(idx)}: {val.item():.2f}", end=", ")

    def __call__(self, *args, k=5, **kwargs):
        assert k <= 100, "Amount must be less than or equal to 100"

        # TODO: be somewhat smarter about this
        args = [
            x
            for arg in args
            for x in ([arg] if not isinstance(arg, (list, tuple)) else arg)
        ]

        for feature in args:
            print(f"feature {feature}")
            input_ids = self.dataset["input_ids"][self.top_activations[:, feature]][:k]
            self.color_input_ids(input_ids, feature, **kwargs)
            print()
