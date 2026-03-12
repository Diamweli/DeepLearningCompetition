#!/usr/bin/env python3
"""
evaluate_all.py — Évaluation locale des soumissions (.txt et .enc)

Scanne le dossier submissions/ et évalue chaque soumission contre les
vrais labels de test, puis met à jour leaderboard/leaderboard.csv.

Usage:
    python evaluate_all.py --labels chemin/vers/test_labels.npy

Pour les fichiers .enc, la variable d'environnement PRIVATE_KEY doit
être définie (clé PEM en texte ou base64).
"""

import argparse
import csv
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np


def load_predictions(path: str) -> np.ndarray:
    """Charge un fichier de prédictions (.txt ou .npy)."""
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, dtype=int)


def decrypt_submission(enc_path: str, output_path: str) -> bool:
    """Déchiffre un fichier .enc en utilisant encryption/decrypt.py."""
    try:
        from encryption.decrypt import decrypt_file
        decrypt_file(enc_path, output_path)
        return True
    except Exception as exc:
        print(f"  ⚠ Impossible de déchiffrer {enc_path}: {exc}")
        return False


def extract_team_name(filename: str) -> str:
    """Extrait le nom d'équipe du nom de fichier."""
    base = os.path.basename(filename)
    for ext in (".enc", ".txt", ".npy"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def compute_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Calcule l'accuracy (taux de bonnes prédictions)."""
    if len(preds) != len(y_true):
        raise ValueError(
            f"Nombre de prédictions ({len(preds)}) ≠ nombre de labels ({len(y_true)})"
        )
    return float(np.mean(preds == y_true))


def load_leaderboard(path: str) -> list:
    """Charge le leaderboard existant depuis un CSV."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_leaderboard(path: str, rows: list):
    """Sauvegarde le leaderboard trié dans le CSV."""
    # Trier par accuracy décroissante
    rows.sort(key=lambda r: float(r.get("accuracy", 0)), reverse=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "accuracy", "submitted_at"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "team": row["team"],
                    "accuracy": f"{float(row['accuracy']):.6f}",
                    "submitted_at": row.get("submitted_at", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Évalue toutes les soumissions et met à jour le leaderboard."
    )
    parser.add_argument(
        "--labels",
        default=os.environ.get("TEST_LABELS_PATH", "evaluation/test_labels.npy"),
        help="Chemin vers le fichier test_labels.npy (default: evaluation/test_labels.npy)",
    )
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Dossier contenant les soumissions (default: submissions/)",
    )
    parser.add_argument(
        "--leaderboard",
        default="leaderboard/leaderboard.csv",
        help="Chemin du fichier leaderboard CSV (default: leaderboard/leaderboard.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Réévaluer même les équipes déjà présentes dans le leaderboard",
    )
    args = parser.parse_args()

    # Vérifier que les labels existent
    if not os.path.exists(args.labels):
        print(f"❌ Fichier de labels introuvable: {args.labels}")
        print("   Spécifiez le chemin avec --labels ou la variable TEST_LABELS_PATH")
        sys.exit(1)

    y_true = np.load(args.labels)
    print(f"✅ Labels chargés: {len(y_true)} exemples\n")

    # Charger le leaderboard existant
    existing = load_leaderboard(args.leaderboard)
    existing_teams = {row["team"] for row in existing}

    # Scanner les soumissions
    if not os.path.isdir(args.submissions_dir):
        print(f"❌ Dossier introuvable: {args.submissions_dir}")
        sys.exit(1)

    files = sorted(os.listdir(args.submissions_dir))
    submission_files = [
        f
        for f in files
        if f.endswith(".txt") or f.endswith(".enc") or f.endswith(".npy")
    ]

    if not submission_files:
        print(f"⚠ Aucune soumission trouvée dans {args.submissions_dir}/")
        sys.exit(0)

    print(f"📂 {len(submission_files)} soumission(s) trouvée(s):\n")

    new_entries = []
    for filename in submission_files:
        team = extract_team_name(filename)
        filepath = os.path.join(args.submissions_dir, filename)

        # Ignorer si déjà évalué (sauf si --force)
        if team in existing_teams and not args.force:
            print(f"  ⏭ {team:20s} — déjà dans le leaderboard, ignoré")
            continue

        print(f"  📊 {team:20s} — ", end="")

        try:
            if filename.endswith(".enc"):
                # Déchiffrer d'abord
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                    tmp_path = tmp.name
                if not decrypt_submission(filepath, tmp_path):
                    continue
                preds = load_predictions(tmp_path)
                os.unlink(tmp_path)
            else:
                preds = load_predictions(filepath)

            accuracy = compute_accuracy(preds, y_true)
            print(f"accuracy = {accuracy:.6f} ({accuracy * 100:.2f}%)")

            new_entries.append(
                {
                    "team": team,
                    "accuracy": accuracy,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        except Exception as exc:
            print(f"❌ Erreur: {exc}")

    # Mettre à jour le leaderboard
    if new_entries:
        # Si --force, supprimer les entrées existantes pour les équipes réévaluées
        if args.force:
            new_team_names = {e["team"] for e in new_entries}
            existing = [r for r in existing if r["team"] not in new_team_names]

        all_rows = existing + new_entries
        save_leaderboard(args.leaderboard, all_rows)
        print(f"\n✅ Leaderboard mis à jour: {len(all_rows)} équipe(s) au total")
        print(f"   → {args.leaderboard}")
    else:
        print("\nℹ Aucune nouvelle entrée à ajouter.")

    # Afficher le classement final
    final = load_leaderboard(args.leaderboard)
    final.sort(key=lambda r: float(r.get("accuracy", 0)), reverse=True)
    print("\n" + "=" * 60)
    print(f"{'RANG':>5}  {'ÉQUIPE':20s}  {'SCORE':>12}")
    print("-" * 60)
    for i, row in enumerate(final, 1):
        score = float(row.get("accuracy", 0))
        print(f"{i:>5}  {row['team']:20s}  {score * 100:>10.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
