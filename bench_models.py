#!/usr/bin/env python3
"""Model backends for the storage / dataloader benchmarks.

Both backends consume the same input the loaders already produce: a ``uint8``
batch of base codes with shape ``(batch, window)`` where A=0, C=1, G=2, T=3,
N/unknown=4. Each backend owns its model + optimizer and exposes one training
step, so the benchmark scripts stay model-agnostic.

Backends
--------
``tiny``
    The original two-layer next-base convolution (per base). Cheap; its GPU
    ceiling is very high, so it mainly stresses the storage path.

``nt``
    A Nucleotide-Transformer-style masked-language-model encoder: a BERT/ESM
    transformer over non-overlapping k-mer tokens (NT uses 6-mers), sized to
    the published NT v2 family. It is pure PyTorch, so it needs no
    ``transformers`` install, no weight download, and runs offline on a compute
    node. It reproduces NT's *compute* (the FLOPs that set the GPU ceiling),
    not its pretrained weights or exact tokenizer vocabulary — which is exactly
    what a throughput/starvation benchmark needs.

    Reference: Dalla-Torre et al., "Nucleotide Transformer", Nature Methods
    (2024); models at https://github.com/instadeepai/nucleotide-transformer and
    on the HuggingFace hub under ``InstaDeepAI/nucleotide-transformer-v2-*``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# Approximate configs for the NT v2 family (hidden, layers, heads, ffn). These
# match the published parameter tiers closely enough to reproduce the compute;
# the exact param count is printed at build time so a run is self-documenting.
NT_PRESETS = {
    "50m": dict(hidden=512, layers=12, heads=8, ffn=2048),
    "100m": dict(hidden=640, layers=20, heads=10, ffn=2560),
    "250m": dict(hidden=768, layers=24, heads=12, ffn=3072),
    "500m": dict(hidden=1280, layers=24, heads=20, ffn=5120),
    "2b5": dict(hidden=2560, layers=32, heads=20, ffn=10240),
}


class TinyNextBaseModel(nn.Module):
    """The original per-base next-base convolution."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 16)
        self.network = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 5, kernel_size=1),
        )

    def forward(self, inputs):
        return self.network(self.embedding(inputs).transpose(1, 2))


class NTStyleMaskedLM(nn.Module):
    """BERT/ESM-style masked-LM encoder over k-mer tokens, NT-sized."""

    def __init__(self, kmer=6, hidden=1280, layers=24, heads=20, ffn=5120, mask_prob=0.15):
        super().__init__()
        self.kmer = int(kmer)
        self.num_kmers = 4 ** self.kmer
        self.cls_id = self.num_kmers
        self.mask_id = self.num_kmers + 1
        self.pad_id = self.num_kmers + 2
        self.vocab = self.num_kmers + 3
        self.mask_prob = float(mask_prob)

        self.embed = nn.Embedding(self.vocab, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first (pre-LN, as ESM/NT
        # use); disable it explicitly to avoid a benign per-build warning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, self.vocab)
        # Base-4 place values for packing a k-mer of base codes into one id.
        self.register_buffer("pow4", 4 ** torch.arange(self.kmer - 1, -1, -1), persistent=False)

    def tokenize(self, base_codes: torch.Tensor) -> torch.Tensor:
        """(batch, window) uint8 base codes -> (batch, tokens+1) long ids with a CLS."""
        batch, window = base_codes.shape
        tokens = window // self.kmer
        codes = base_codes[:, : tokens * self.kmer].long().clamp(max=3)  # N(4) -> T for id packing
        codes = codes.view(batch, tokens, self.kmer)
        ids = (codes * self.pow4).sum(dim=-1)  # (batch, tokens) in [0, 4**k)
        cls = torch.full((batch, 1), self.cls_id, device=ids.device, dtype=ids.dtype)
        return torch.cat([cls, ids], dim=1)

    def forward_loss(self, ids: torch.Tensor) -> torch.Tensor:
        labels = ids.clone()
        selected = torch.rand(ids.shape, device=ids.device) < self.mask_prob
        selected[:, 0] = False  # never mask the CLS token
        labels[~selected] = -100
        inputs = ids.clone()
        inputs[selected] = self.mask_id
        hidden = self.embed(inputs)
        hidden = self.encoder(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return F.cross_entropy(logits.reshape(-1, self.vocab), labels.reshape(-1), ignore_index=-100)


class _Backend:
    def __init__(self, model, device, lr, kind, describe):
        self.model = model.to(device)
        self.device = device
        self.kind = kind
        self._describe = describe
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def describe(self) -> str:
        return f"{self._describe} ({self.num_parameters / 1e6:.0f}M params)"

    def synthetic_batch(self, batch_size, window_size) -> torch.Tensor:
        """A device-resident uint8 base-code batch for the compute ceiling."""
        return torch.randint(0, 5, (batch_size, window_size), dtype=torch.uint8, device=self.device)

    def step(self, base_codes: torch.Tensor) -> int:
        """Run one train step on a (batch, window) uint8 base-code batch; return bases processed."""
        batch = base_codes.to(self.device, non_blocking=True)
        if self.kind == "tiny":
            batch = batch.long()
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = self.model(inputs).transpose(1, 2)
            loss = self.criterion(logits.reshape(-1, 5), targets.reshape(-1))
            processed = int(targets.numel())
        else:
            ids = self.model.tokenize(batch)
            loss = self.model.forward_loss(ids)
            processed = int(batch.shape[0] * batch.shape[1])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return processed


def make_backend(kind, window_size, device, *, nt_size="500m", kmer=6, lr=1e-4) -> _Backend:
    """Build a benchmark backend. ``kind`` is 'tiny' or 'nt'."""
    if kind == "tiny":
        return _Backend(TinyNextBaseModel(), device, lr, "tiny", "tiny next-base conv")
    if kind == "nt":
        if nt_size not in NT_PRESETS:
            raise ValueError(f"unknown --nt-size {nt_size!r}; choose from {sorted(NT_PRESETS)}")
        preset = NT_PRESETS[nt_size]
        if window_size < kmer:
            raise ValueError("window_size must be >= kmer for the NT backend")
        model = NTStyleMaskedLM(kmer=kmer, **preset)
        tokens = window_size // kmer + 1
        describe = (f"NT-style MLM {nt_size} (k={kmer}, hidden={preset['hidden']}, "
                    f"layers={preset['layers']}, heads={preset['heads']}, ffn={preset['ffn']}, "
                    f"vocab={model.vocab}, tokens/window={tokens})")
        return _Backend(model, device, lr, "nt", describe)
    raise ValueError(f"unknown model kind {kind!r}; choose 'tiny' or 'nt'")
