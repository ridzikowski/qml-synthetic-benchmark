#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised convolutional AE: jeden CSV 60 x 17 -> latent 3/4/5 na nagranie.

Eksport jest zgodny z historycznym speech_ae_3f.csv i mlp_ae_baseline.py:
  Feature_1,...,Feature_N,target,partition,speaker_id,sample_id
  0.1,...,-0.2,1,fit,0132a06d,one/0132a06d_nohash_0

Wyniki: speech_ae_3f.csv, speech_ae_4f.csv, speech_ae_5f.csv.
Partition: fit (trening), ae_val (dobor epoki/lambda), qnn_val (ocena
  eksploracyjna; to nie jest niezalezny finalny test).

Wejscie: katalog utworzony przez gsc_wav_to_csv_60x17.py, np.:
  google speech command dataset_csv/00. zero/abc_nohash_0.csv
  google speech command dataset_csv/01. one/def_nohash_1.csv

Klasyfikator czyta WYLACZNIE wektor latentny. Encoder jest optymalizowany
funkcja CE_klasyfikacji + lambda * MSE_rekonstrukcji. Lambda jest dobierana
na ae_val (macro-F1, nastepnie CE); qnn_val nie sluzy do wyboru modelu.

Zaleznosci: pip install numpy torch scikit-learn

Przyklad:
  python train_gsc_supervised_ae.py --input "google speech command dataset_csv" \
      --classes zero one --latent-dims 3 4 5 --epochs 50

Reprodukcja starego podzialu dla probek z poprzedniego badania:
  python train_gsc_supervised_ae.py --classes zero one \
      --reference-csv datasets/speech_ae_3f.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROWS = 60
ORIGINAL_FEATURES = 17
FEATURE_COLUMNS = [f"MFCC_{i:02d}" for i in range(1, 13)] + [
    "F0_Hz", "F1_Hz", "F2_Hz", "F3_Hz", "F4_Hz"
]
# Na wejsciu sieci sa 17 cech po normalizacji + 5 flag dostepnosci F0/F1-F4.
MODEL_CHANNELS = 22


@dataclass(frozen=True)
class Sample:
    path: Path
    relative: str
    word: str
    label_id: int
    speaker_id: str

    @property
    def sample_id(self) -> str:
        # Usuwamy tylko prefiks katalogu klasy (np. "00. zero") i rozszerzenie.
        parts = Path(self.relative).parts
        return (Path(self.word, *parts[1:]).with_suffix("")).as_posix()


def word_from_directory(name: str) -> str:
    # '00. zero' -> 'zero'; '00_zero' -> 'zero'; 'zero' -> 'zero'
    return re.sub(r"^\s*\d+\s*[._-]\s*", "", name).strip().casefold()


def speaker_from_filename(stem: str) -> tuple[str, bool]:
    # Google Speech Commands: hash_nohash_0.wav, hash_nohash_1.wav, ...
    match = re.match(r"^(.+?)_nohash_\d+$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).casefold(), True
    # Dla rozszerzen zbioru, np. '001M_cat.wav' -> '001m'.
    if "_" in stem:
        return stem.split("_", 1)[0].casefold(), False
    return stem.casefold(), False


def available_classes(root: Path) -> dict[str, Path]:
    result = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if not any(p.is_file() and p.suffix.lower() == ".csv" for p in directory.rglob("*")):
            continue
        word = word_from_directory(directory.name)
        if word in result:
            raise ValueError(f"Dwie lokalizacje dla klasy {word!r}: {result[word]}, {directory}")
        result[word] = directory
    return result


def inventory(root: Path, selected: list[str], limit: int, seed: int) -> tuple[list[Sample], list[str]]:
    all_classes = available_classes(root)
    if not all_classes:
        raise ValueError(f"Nie znaleziono katalogow ze zbiorami CSV w {root}")
    words = [word_from_directory(name) for name in selected]
    if len(set(words)) != len(words):
        raise ValueError("Powtorzono nazwy klas w --classes")
    unknown = sorted(set(words) - set(all_classes))
    if unknown:
        raise ValueError(
            f"Nie znaleziono klas: {unknown}. Dostepne: {', '.join(sorted(all_classes))}"
        )
    if len(words) < 2:
        raise ValueError("Wybierz co najmniej dwie klasy")
    # Zgodnosc z historycznym MLP: 0=zero, 1=one niezaleznie od kolejnosci CLI.
    if set(words) == {"zero", "one"}:
        words = ["zero", "one"]

    rng = random.Random(seed)
    result: list[Sample] = []
    n_standard = 0
    for label_id, word in enumerate(words):
        folder = all_classes[word]
        paths = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"),
                       key=lambda p: p.as_posix().casefold())
        if limit > 0 and len(paths) > limit:
            paths = sorted(rng.sample(paths, limit))
        if not paths:
            raise ValueError(f"Klasa {word} nie ma plikow CSV")
        for p in paths:
            speaker, standard = speaker_from_filename(p.stem)
            n_standard += int(standard)
            result.append(Sample(p, p.relative_to(root).as_posix(), word, label_id, speaker))
        print(f"  {word:>15}: {len(paths):>6} plikow")
    sample_ids = [s.sample_id for s in result]
    duplicates = [key for key, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Niejednoznaczne sample_id: {duplicates[:5]}")
    if n_standard < len(result):
        warnings.warn(
            f"Tylko {n_standard}/{len(result)} nazw posiada standardowy wzorzec "
            "'_nohash_'. Dla pozostalych identyfikator mowcy jest heurystyczny; "
            "zweryfikuj rozlacznosc mowcow w split_manifest.csv.", stacklevel=2
        )
    return result, words


def read_feature_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        try:
            header = next(csv.reader(f))
        except StopIteration:
            raise ValueError(f"Pusty CSV: {path}") from None
    if header != FEATURE_COLUMNS:
        raise ValueError(f"Bledny naglowek w {path}: {header!r}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if arr.shape != (ROWS, ORIGINAL_FEATURES):
        raise ValueError(f"W {path}: {arr.shape}, oczekiwano {(ROWS, ORIGINAL_FEATURES)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf w {path}")
    return arr


def choose_group_holdout(y: np.ndarray, groups: np.ndarray, ratio: float,
                         seed: int, tries: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Wybiera podzial bez nakladajacych sie mowcow i ze wszystkimi klasami."""
    classes = int(y.max()) + 1
    global_distribution = np.bincount(y, minlength=classes) / len(y)
    splitter = GroupShuffleSplit(n_splits=tries, test_size=ratio, random_state=seed)
    chosen = None
    best_score = float("inf")
    for train_idx, hold_idx in splitter.split(np.zeros(len(y)), y, groups):
        train_counts = np.bincount(y[train_idx], minlength=classes)
        hold_counts = np.bincount(y[hold_idx], minlength=classes)
        if np.any(train_counts == 0) or np.any(hold_counts == 0):
            continue
        hold_distribution = hold_counts / hold_counts.sum()
        # Bliskosc do zadanej proporcji i globalnego rozkladu klas.
        score = (3.0 * abs(len(hold_idx) / len(y) - ratio)
                 + np.abs(hold_distribution - global_distribution).sum())
        if score < best_score:
            best_score = score
            chosen = (np.asarray(train_idx), np.asarray(hold_idx))
    if chosen is None:
        raise ValueError(
            "Nie udalo sie utworzyc rozlacznego podzialu po mowcach z kazda klasa "
            "w obu czesciach. Potrzeba wiecej roznych mowcow/przykladow na klase."
        )
    return chosen


def make_splits(samples: list[Sample], seed: int, test_fraction: float,
                val_fraction: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    y = np.asarray([s.label_id for s in samples], dtype=np.int64)
    groups = np.asarray([s.speaker_id for s in samples], dtype=str)
    remain, test = choose_group_holdout(y, groups, test_fraction, seed)
    # Walidacja: frakcja wzgledem pierwotnej liczby probek.
    remain_train, val_local = choose_group_holdout(
        y[remain], groups[remain], val_fraction / (1.0 - test_fraction), seed + 1
    )
    split = {"train": np.sort(remain[remain_train]),
             "val": np.sort(remain[val_local]), "test": np.sort(test)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(groups[split[a]]).isdisjoint(set(groups[split[b]])), (a, b)
    assert len(np.unique(np.concatenate(list(split.values())))) == len(samples)
    for name, idx in split.items():
        counts = np.bincount(y[idx], minlength=int(y.max()) + 1)
        print(f"  {name:>5}: {len(idx):>6} probek, "
              f"{len(set(groups[idx])):>5} mowcow, klasy {counts.tolist()}")
    return split, y


def make_reference_splits(samples: list[Sample], words: list[str],
                          reference_path: Path) -> tuple[list[Sample], dict[str, np.ndarray], np.ndarray]:
    """Odtwarza historyczne fit/ae_val/qnn_val na dokladnie tych samych probkach.

    Celowo NIE doklada nagran bez odpowiednika w referencji: ich dolaczenie
    zmieniloby eksperyment porownawczy. Brak probek z referencji to blad.
    """
    ref: dict[str, dict[str, str]] = {}
    reference_speakers: dict[str, str] = {}
    valid_partitions = {"fit", "ae_val", "qnn_val"}
    with reference_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "target", "partition", "speaker_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Niepoprawny --reference-csv, wymagane {sorted(required)}")
        for row in reader:
            sample_id = str(row["sample_id"]).strip()
            partition = str(row["partition"]).strip()
            speaker = str(row["speaker_id"]).strip().casefold()
            if partition not in valid_partitions:
                raise ValueError(f"Nieznana partition dla {sample_id}: {partition}")
            if not speaker or not sample_id or sample_id in ref:
                raise ValueError(f"Pusty/zdublowany identyfikator w referencji: {sample_id}")
            if speaker in reference_speakers and reference_speakers[speaker] != partition:
                raise ValueError(f"Przeciek mowcy w pliku referencyjnym: {speaker}")
            reference_speakers[speaker] = partition
            ref[sample_id] = {"partition": partition, "speaker_id": speaker,
                              "target": str(row["target"]).strip()}
    by_id = {s.sample_id: s for s in samples}
    selected_word_set = set(words)
    wanted = {sid: row for sid, row in ref.items()
              if sid.split("/", 1)[0].casefold() in selected_word_set}
    absent = sorted(set(wanted) - set(by_id))
    if absent:
        raise ValueError(f"Brakuje {len(absent)} plikow z historycznego podzialu, np. {absent[:4]}. "
                         "Nie mozna odtworzyc dokladnie tego samego eksperymentu.")
    if not wanted:
        raise ValueError("Brak wspolnych klas pomiedzy --classes i --reference-csv")
    selected_samples = [s for s in samples if s.sample_id in wanted]
    mapping = {"fit": "train", "ae_val": "val", "qnn_val": "test"}
    partitions = {"train": [], "val": [], "test": []}
    for i, s in enumerate(selected_samples):
        meta = wanted[s.sample_id]
        if s.speaker_id != meta["speaker_id"]:
            raise ValueError(f"Niezgodny speaker_id: {s.sample_id}")
        if str(s.label_id) != meta["target"]:
            raise ValueError(f"Niezgodny target dla {s.sample_id}: {s.label_id} != {meta['target']}; "
                             "podaj --classes w kolejnosci historycznych etykiet")
        partitions[mapping[meta["partition"]]].append(i)
    splits = {name: np.asarray(idx, dtype=np.int64) for name, idx in partitions.items()}
    y = np.asarray([s.label_id for s in selected_samples], dtype=np.int64)
    for name, idx in splits.items():
        if len(idx) == 0 or set(y[idx]) != set(range(len(words))):
            raise ValueError(f"Historyczny podzial {name} nie zawiera wszystkich klas")
    print(f"Odtworzono podzial z {reference_path}: {len(selected_samples)} probek")
    for name, idx in splits.items():
        print(f"  {name}: {len(idx)} probek, {len({selected_samples[i].speaker_id for i in idx})} mowcow")
    return selected_samples, splits, y


class AcousticScaler:
    """Statystyki tylko na TRAIN; zera F0/F1-F4 sa kodami brakow (osobne maski)."""

    def __init__(self) -> None:
        self.mean = np.zeros(ORIGINAL_FEATURES, dtype=np.float32)
        self.std = np.ones(ORIGINAL_FEATURES, dtype=np.float32)

    def fit(self, x_train: np.ndarray) -> "AcousticScaler":
        for j in range(ORIGINAL_FEATURES):
            v = x_train[:, :, j].reshape(-1)
            if j >= 12:
                v = v[v > 0.0]  # zera = brak F0 / estymaty formantu
            if len(v):
                self.mean[j] = float(v.mean())
                self.std[j] = max(float(v.std()), 1e-5)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        standardized = (x - self.mean[None, None, :]) / self.std[None, None, :]
        valid = (x[:, :, 12:] > 0.0).astype(np.float32)
        standardized[:, :, 12:] *= valid  # w brakujacym polu zawsze 0 po normalizacji
        standardized = np.clip(standardized, -8.0, 8.0)
        return np.concatenate([standardized, valid], axis=2).transpose(0, 2, 1).astype(np.float32)

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "description": "12 MFCC: z-score; F0/F1-F4: train z-score po wartosciach >0; "
                               "zera wypelnione 0 + piec mask obecnosci; clip [-8,8]"}


class SupervisedConvAE(nn.Module):
    """[B,22,60] -> [B,latent_dim]; decoder + klasyfikator uzywaja tylko z."""

    def __init__(self, latent_dim: int, n_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(MODEL_CHANNELS, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),  # 60 -> 30
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),  # 30 -> 15
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 15, 128), nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 15), nn.GELU(),
            nn.Unflatten(1, (64, 15)),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(32, MODEL_CHANNELS, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, n_classes)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_head(self.feature_extractor(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z), self.classifier(z)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(y[indices]))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0)


def evaluate(model: SupervisedConvAE, loader: DataLoader,
             device: torch.device, classes: int) -> dict:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    total_ce, total_rec, total = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, reconstruction, logits = model(x)
            size = y.numel()
            total += size
            total_ce += float(F.cross_entropy(logits, y, reduction="sum").item())
            total_rec += float(F.mse_loss(reconstruction, x, reduction="mean").item()) * size
            all_true.extend(y.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
    return {"accuracy": float(accuracy_score(all_true, all_pred)),
            "macro_f1": float(f1_score(all_true, all_pred, labels=list(range(classes)),
                                       average="macro", zero_division=0)),
            "cross_entropy": total_ce / total,
            "reconstruction_mse": total_rec / total,
            "confusion_matrix": confusion_matrix(all_true, all_pred,
                                                  labels=list(range(classes))).tolist()}


def is_better(new: dict, old: dict | None) -> bool:
    # Pierwsze kryterium macro-F1 (klasyfikacja); CE rozstrzyga remisy.
    if old is None:
        return True
    if new["macro_f1"] > old["macro_f1"] + 1e-5:
        return True
    if abs(new["macro_f1"] - old["macro_f1"]) <= 1e-5:
        return new["cross_entropy"] < old["cross_entropy"] - 1e-5
    return False


def fit_candidate(latent_dim: int, recon_weight: float,
                  train_loader: DataLoader, val_loader: DataLoader,
                  train_labels: np.ndarray, n_classes: int,
                  device: torch.device, epochs: int, patience: int,
                  lr: float, seed: int) -> tuple[SupervisedConvAE, dict, list[dict], int]:
    # Te same wagi startowe dla roznych wag rekonstrukcji przy tym samym latent_dim.
    set_seed(seed + latent_dim * 1000)
    model = SupervisedConvAE(latent_dim, n_classes).to(device)
    counts = np.bincount(train_labels, minlength=n_classes)
    weights = np.sqrt(counts.sum() / (n_classes * counts.astype(np.float32)))
    weights /= np.mean(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_metrics = None
    best_state = None
    best_epoch = 0
    history = []
    no_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_ce, train_rec, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, reconstruction, logits = model(x)
            ce = F.cross_entropy(logits, y, weight=class_weights)
            rec = F.mse_loss(reconstruction, x)
            loss = ce + recon_weight * rec
            if not torch.isfinite(loss):
                raise FloatingPointError("Niestabilny loss (NaN/Inf)")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            n += y.numel()
            train_ce += ce.item() * y.numel()
            train_rec += rec.item() * y.numel()
        val_metrics = evaluate(model, val_loader, device, n_classes)
        scheduler.step(val_metrics["cross_entropy"])
        history.append({"latent_dim": latent_dim, "recon_weight": recon_weight,
                        "epoch": epoch, "train_weighted_ce": train_ce / n,
                        "train_mse": train_rec / n,
                        "val_ce": val_metrics["cross_entropy"],
                        "val_mse": val_metrics["reconstruction_mse"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"]})
        if is_better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 5 == 0 or no_improvement >= patience:
            print(f"    ep {epoch:>3}: val acc={val_metrics['accuracy']:.3f}, "
                  f"F1={val_metrics['macro_f1']:.3f}, CE={val_metrics['cross_entropy']:.3f}")
        if no_improvement >= patience:
            break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, history, best_epoch


@torch.no_grad()
def encode_all(model: SupervisedConvAE, features: np.ndarray,
               device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    latents = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        latents.append(model.encode(x).cpu().numpy())
    return np.concatenate(latents, axis=0)


def write_compatible_csv(output_file: Path, values: np.ndarray,
                         samples: list[Sample], splits: dict[str, np.ndarray]) -> None:
    """Format 1:1 zgodny z mlp_ae_baseline.py (nazwy i kolejnosc kolumn)."""
    assert values.shape == (len(samples), values.shape[1])
    if not np.isfinite(values).all():
        raise ValueError(f"NaN/Inf w latentach przeznaczonych do {output_file}")
    split_names = {"train": "fit", "val": "ae_val", "test": "qnn_val"}
    assigned: dict[int, str] = {}
    for internal, indices in splits.items():
        assigned.update({int(i): split_names[internal] for i in indices})
    if len(assigned) != len(samples):
        raise ValueError("Nie wszystkie probki maja partition")

    order = {"fit": 0, "ae_val": 1, "qnn_val": 2}
    row_order = sorted(range(len(samples)),
                       key=lambda i: (order[assigned[i]], samples[i].sample_id.casefold()))
    header = [f"Feature_{i + 1}" for i in range(values.shape[1])]
    header += ["target", "partition", "speaker_id", "sample_id"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in row_order:
            sample = samples[i]
            writer.writerow([f"{float(v):.9g}" for v in values[i]] +
                            [sample.label_id, assigned[i], sample.speaker_id, sample.sample_id])
    print(f"  CSV: {output_file} ({len(row_order)} wierszy; {len(header)} kolumn)")


def write_manifest(output: Path, samples: list[Sample], splits: dict[str, np.ndarray]) -> None:
    assigned = {}
    split_names = {"train": "fit", "val": "ae_val", "test": "qnn_val"}
    for split_name, indices in splits.items():
        assigned.update({int(i): split_names[split_name] for i in indices})
    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "target", "partition", "speaker_id", "source_csv"])
        for i, sample in enumerate(samples):
            writer.writerow([sample.sample_id, sample.label_id, assigned[i],
                             sample.speaker_id, sample.relative])


def write_history(output: Path, history: list[dict]) -> None:
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised AE: sekwencje 60x17 -> latenty 3/4/5")
    parser.add_argument("--input", type=Path, default=Path("google speech command dataset_csv"))
    parser.add_argument("--output", type=Path, default=Path("out/gsc_supervised_ae"))
    parser.add_argument("--classes", nargs="+", help="Np. zero one, bed bird cat dog down")
    parser.add_argument("--list-classes", action="store_true", help="Wypisz znalezione klasy i zakoncz")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--recon-weights", type=float, nargs="+", default=[0.05, 0.20],
                        help="Szukane wagi lambda dla CE + lambda*MSE")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 = brak limitu")
    parser.add_argument("--reference-csv", type=Path, default=None,
                        help="Opcjonalnie: zachowaj DOKLADNIE historyczne sample_id/partition "
                             "z pliku speech_ae_3f.csv; pomija dodatkowe nagrania")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--export-angles", action="store_true",
                        help="Dodatkowo zapisz latenty skalowane do [0, 2*pi] wg TRAIN, z clippingiem")
    args = parser.parse_args()
    if not args.list_classes and not args.classes:
        parser.error("Podaj --classes albo --list-classes")
    if len(set(args.latent_dims)) != len(args.latent_dims) or any(d < 1 for d in args.latent_dims):
        parser.error("--latent-dims: dodatnie, niepowtarzajace sie liczby")
    if any(v <= 0 for v in args.recon_weights) or not args.recon_weights:
        parser.error("--recon-weights: wartosci dodatnie")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs/patience/batch-size/lr musza byc dodatnie")
    if not (0 < args.val_fraction < 0.5 and 0 < args.test_fraction < 0.5
            and args.val_fraction + args.test_fraction < 0.7):
        parser.error("Wymagane: val i test w (0,0.5), suma < 0.7")
    if args.limit_per_class < 0:
        parser.error("--limit-per-class musi byc >= 0")
    return args


def main() -> None:
    args = cli()
    root = args.input.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Brak katalogu {root}. Najpierw wygeneruj CSV z WAV.")
    if args.list_classes:
        print("Dostepne klasy:")
        for word, directory in sorted(available_classes(root).items()):
            print(f"  {word:<16} ({directory.name})")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Wybrano --device cuda, ale PyTorch nie widzi CUDA")
    device = torch.device("cuda" if (args.device == "cuda" or
                                    (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    set_seed(args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Wybrane klasy:")
    samples, words = inventory(root, args.classes, args.limit_per_class, args.seed)
    print(f"Lacznie {len(samples)} nagran | urzadzenie: {device}")
    if args.reference_csv is not None:
        samples, splits, y = make_reference_splits(
            samples, words, args.reference_csv.expanduser().resolve()
        )
    else:
        print("Podzial po mowcach:")
        splits, y = make_splits(samples, args.seed, args.test_fraction, args.val_fraction)
    write_manifest(output, samples, splits)
    (output / "class_map.json").write_text(
        json.dumps({word: i for i, word in enumerate(words)}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Odczyt macierzy 60 x 17 ...")
    matrices = np.stack([read_feature_csv(s.path) for s in samples], axis=0)
    scaler = AcousticScaler().fit(matrices[splits["train"]])
    x = scaler.transform(matrices)
    del matrices  # nie przechowuj dwoch reprezentacji w RAM
    print(f"Wejscie sieci: {x.shape[1:]} (17 cech + 5 masek) | "
          f"statystyki liczone tylko na train")
    train_loader = make_loader(x, y, splits["train"], args.batch_size, True, args.seed)
    val_loader = make_loader(x, y, splits["val"], args.batch_size, False, args.seed)
    test_loader = make_loader(x, y, splits["test"], args.batch_size, False, args.seed)
    results = {"classes": words, "input_dir": str(root), "seed": args.seed,
               "partitions": {"fit": len(splits["train"]),
                              "ae_val": len(splits["val"]),
                              "qnn_val": len(splits["test"])},
               "reference_csv": str(args.reference_csv) if args.reference_csv else None,
               "objective": "fit weighted CE(classifier(encoder(X))) + lambda * MSE(decoder(encoder(X)), X)",
               "selection": "ae_val macro-F1 then ae_val CE; qnn_val exploratory only",
               "note": "qnn_val nie jest niezaleznym finalnym testem",
               "scaler": scaler.as_dict(), "by_dim": {}}
    full_history: list[dict] = []
    for dim in args.latent_dims:
        print(f"\n=== LATENT DIM = {dim} ===")
        best_candidate = None
        best_val = None
        best_lambda = None
        best_epoch = None
        for weight in args.recon_weights:
            print(f"  Proba lambda={weight:g}")
            model, val_metrics, history, epoch = fit_candidate(
                dim, weight, train_loader, val_loader, y[splits["train"]],
                len(words), device, args.epochs, args.patience, args.lr, args.seed
            )
            full_history.extend(history)
            print(f"  -> najlepsza epoka {epoch}, F1={val_metrics['macro_f1']:.4f}, "
                  f"accuracy={val_metrics['accuracy']:.4f}")
            if is_better(val_metrics, best_val):
                best_candidate = model
                best_val = val_metrics
                best_lambda = weight
                best_epoch = epoch
            else:
                del model
        assert best_candidate is not None and best_val is not None
        # qnn_val nie sluzy do wyboru lambda/epoki; to wynik eksploracyjny.
        qnn_val_metrics = evaluate(best_candidate, test_loader, device, len(words))
        latents = encode_all(best_candidate, x, device, args.batch_size)
        write_compatible_csv(output / f"speech_ae_{dim}f.csv", latents, samples, splits)
        checkpoint = {
            "state_dict": {k: v.detach().cpu() for k, v in best_candidate.state_dict().items()},
            "latent_dim": dim, "num_classes": len(words), "classes": words,
            "input_channels": MODEL_CHANNELS, "rows": ROWS,
            "recon_weight": best_lambda, "best_epoch": best_epoch,
            "seed": args.seed, "scaler": scaler.as_dict(),
            "training_speakers": sorted({samples[i].speaker_id for i in splits["train"]}),
        }
        torch.save(checkpoint, output / f"model_latents_{dim}.pt")
        if args.export_angles:
            train_z = latents[splits["train"]]
            lo = train_z.min(axis=0)
            hi = train_z.max(axis=0)
            span = np.where(hi - lo < 1e-8, 1.0, hi - lo)
            angles = np.clip((latents - lo) / span, 0.0, 1.0) * (2.0 * np.pi)
            write_compatible_csv(output / f"speech_ae_{dim}f_angles.csv", angles, samples, splits)
            checkpoint["angles_train_min"] = lo.tolist()
            checkpoint["angles_train_max"] = hi.tolist()
            torch.save(checkpoint, output / f"model_latents_{dim}.pt")
        results["by_dim"][str(dim)] = {
            "best_recon_weight": best_lambda,
            "best_epoch": best_epoch,
            "validation": best_val,
            "qnn_val_exploratory": qnn_val_metrics,
            "model_file": f"model_latents_{dim}.pt",
        }
        print(f"WYBRANO lambda={best_lambda:g}, epoka={best_epoch}. "
              f"VAL F1={best_val['macro_f1']:.4f}, "
              f"qnn_val eksploracyjnie acc={qnn_val_metrics['accuracy']:.4f}, "
              f"macro-F1={qnn_val_metrics['macro_f1']:.4f}")
        del best_candidate

    write_history(output, full_history)
    # Porownanie wymiarow jedynie wg walidacji, NIE wg testu.
    top_dim = max(args.latent_dims, key=lambda dim: (
        results["by_dim"][str(dim)]["validation"]["macro_f1"],
        -results["by_dim"][str(dim)]["validation"]["cross_entropy"]
    ))
    results["recommended_dim_by_validation"] = top_dim
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nGotowe. Pliki: {output}")
    print(f"Preferowany rozmiar wg VAL: {top_dim} | pelne metryki: metrics.json")
    print("Kazdy speech_ae_Nf.csv zawiera wszystkie trzy partycje: fit, ae_val, qnn_val.")
    if len(words) == 2 and words == ["zero", "one"] and 3 in args.latent_dims:
        print("Oryginalny MLP (zero/one): python mlp_ae_baseline.py --csv "
              f"\"{output / 'speech_ae_3f.csv'}\"")
    elif len(words) > 2:
        print("Format CSV jest zgodny, ale dolaczony mlp_ae_baseline.py "
              "wymaga adaptacji do klasyfikacji wiecej niz 2 klas.")
    print("Nie zmieniaj podzialu; qnn_val jest zbiorem eksploracyjnym, "
          "nie niezaleznym finalnym testem.")


if __name__ == "__main__":
    main()
