# -*- coding: utf-8 -*-
"""
Dataset không nhãn cho CNSCED-main, trả về (waveform, padding_mask)
"""
import os, glob, torchaudio, torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class CNSCEDUnlabeled(Dataset):
    """
    Đọc tất cả .wav trong datasets/CNSCED-main/F và M
    Trả về:
      wav:  Tensor (T,)
      pad:  Bool Tensor (T,) với True đánh dấu các vị trí padding
    """
    def __init__(self, root_dir="datasets/CNSCED-main", sample_rate=16000):
        self.sr = sample_rate
        # gom tất cả wav trong F/, M/
        self.wav_paths: List[str] = []
        for sub in ("F","M"):
            self.wav_paths += glob.glob(
                os.path.join(root_dir, sub, "**", "*.wav"),
                recursive=True
            )
        if not self.wav_paths:
            raise RuntimeError(f"Không tìm thấy file .wav trong {root_dir}")
        self.wav_paths.sort()

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.wav_paths[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.squeeze(0)  # (T,)
        pad = torch.zeros_like(wav, dtype=torch.bool)
        return wav, pad

def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pad thành batch đồng chiều dài
    """
    waves, pads = zip(*batch)
    lengths = [w.shape[0] for w in waves]
    maxlen = max(lengths)
    out_w, out_p = [], []
    for w,p in zip(waves, pads):
        L = w.shape[0]
        if L < maxlen:
            diff = maxlen - L
            w = torch.cat([w, w.new_zeros(diff)], dim=0)
            p = torch.cat([p, torch.ones(diff, dtype=torch.bool, device=p.device)])
        out_w.append(w)
        out_p.append(p)
    return torch.stack(out_w), torch.stack(out_p)

def get_loader(batch_size=8, num_workers=4, **kwargs) -> DataLoader:
    ds = CNSCEDUnlabeled(**kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )

if __name__ == "__main__":
    # Demo
    loader = get_loader(batch_size=2)
    w,p = next(iter(loader))
    print("Demo loader:", w.shape, p.shape)
