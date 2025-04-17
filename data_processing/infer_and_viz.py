# -*- coding: utf-8 -*-
"""
1) Kiểm tra checkpoint load được
2) Suy diễn embedding cho toàn bộ wav trong datasets/CNSCED-main
3) Vẽ UMAP 2‑d theo nhãn thư mục (F vs M)
"""
import os, glob, argparse, copy
import numpy as np
import torch, torchaudio
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm

# ** Chỉnh đây **
from data_processing.dataset_tonal import get_loader

# ** Giữ nguyên **
from upstream.emotion2vec_tonal.config_tonal import Data2VecMultiTonalConfig
from upstream.emotion2vec_tonal.emotion2vec_tonal_model import Data2VecMultiTonalCrossAttnModel

def load_model(ckpt_path, device):
    cfg   = Data2VecMultiTonalConfig()
    model = Data2VecMultiTonalCrossAttnModel(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> Load ckpt '{ckpt_path}': missing={len(missing)}, unexpected={len(unexpected)}")
    # tạo teacher copy
    model.teacher = copy.deepcopy(model)
    for p in model.teacher.parameters(): p.requires_grad_(False)
    return model.to(device).eval()

def compute_pitch_zero(wav_tensor):
    # placeholder, thay F0 thật nếu cần
    return torch.zeros_like(wav_tensor)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--dataset",    default="datasets/CNSCED-main")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_neighbors",type=int, default=30)
    p.add_argument("--min_dist",   type=float, default=0.1)
    p.add_argument("--umap_out",   default="umap.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt, device)

    # 1) Test nhanh 1 batch
    loader = get_loader(batch_size=args.batch_size,
                        root_dir=args.dataset)
    wav, pad = next(iter(loader))
    wav, pad = wav.to(device), pad.to(device)
    pitch = compute_pitch_zero(wav)
    with torch.no_grad():
        out = model.extract_features(wav, pitch, pad, mask=False)
    print(">>> Test batch embedding:", out["x"].shape)

    # 2) Infer all
    wav_paths = sorted(glob.glob(os.path.join(args.dataset, "*","**","*.wav"),
                                 recursive=True))
    embs, labs = [], []
    for path in tqdm(wav_paths, desc="Infer all"):
        wav, sr = torchaudio.load(path)
        if sr!=16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0).to(device)
        pad  = torch.zeros_like(wav, dtype=torch.bool, device=device)
        pitch= compute_pitch_zero(wav)
        with torch.no_grad():
            feat = model.extract_features(
                wav.unsqueeze(0), pitch.unsqueeze(0), pad.unsqueeze(0),
                mask=False
            )["x"]  # (1,T,D)
        utt_emb = feat.mean(1).squeeze(0).cpu().numpy()
        embs.append(utt_emb)
        labs.append(os.path.basename(os.path.dirname(path)))

    X   = np.vstack(embs)
    labs= np.array(labs)

    # 3) Vẽ UMAP
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine"
    )
    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(6,6))
    colors={"F":"tab:red","M":"tab:blue"}
    for lbl in np.unique(labs):
        idx = (labs==lbl)
        plt.scatter(X2[idx,0], X2[idx,1],
                    c=colors.get(lbl,"k"),
                    label=lbl, s=10, alpha=0.7)
    plt.legend(); plt.axis("off"); plt.tight_layout()
    plt.savefig(args.umap_out, dpi=300)
    print("Saved UMAP to", args.umap_out)

if __name__=="__main__":
    main()
