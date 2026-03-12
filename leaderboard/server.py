#!/usr/bin/env python3
"""
server.py — Serveur local pour le leaderboard avec upload et évaluation.

Lance un serveur web qui :
  - Sert les fichiers statiques du leaderboard
  - Fournit une API /api/upload pour charger et évaluer des prédictions
  - Met à jour leaderboard.csv automatiquement

Usage:
    python leaderboard/server.py --labels .tmp/test_labels.npy [--port 8000]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np

# Chemin par défaut des fichiers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_CSV = os.path.join(SCRIPT_DIR, "leaderboard.csv")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "submissions")

# Labels de test (chargés au démarrage)
TEST_LABELS = None


def load_predictions_from_bytes(data: bytes) -> np.ndarray:
    """Parse un fichier de prédictions (un entier par ligne)."""
    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    preds = []
    for line in lines:
        try:
            preds.append(int(line))
        except ValueError:
            try:
                preds.append(int(float(line)))
            except ValueError:
                raise ValueError(f"Ligne invalide dans le fichier: '{line}'")
    return np.array(preds, dtype=int)


def compute_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Calcule l'accuracy."""
    if len(preds) != len(y_true):
        raise ValueError(
            f"Nombre de prédictions ({len(preds)}) != nombre de labels attendus ({len(y_true)}). "
            f"Le fichier doit contenir exactement {len(y_true)} prédictions."
        )
    return float(np.mean(preds == y_true))


def load_leaderboard() -> list:
    """Charge le leaderboard CSV existant."""
    if not os.path.exists(LEADERBOARD_CSV):
        return []
    rows = []
    with open(LEADERBOARD_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_leaderboard(rows: list):
    """Sauvegarde le leaderboard CSV trié par score décroissant."""
    rows.sort(key=lambda r: float(r.get("accuracy", 0)), reverse=True)
    with open(LEADERBOARD_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "accuracy", "submitted_at"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "team": row["team"],
                "accuracy": f"{float(row['accuracy']):.6f}",
                "submitted_at": row.get("submitted_at", ""),
            })


def add_to_leaderboard(team: str, accuracy: float) -> list:
    """Ajoute ou met à jour une équipe dans le leaderboard."""
    rows = load_leaderboard()
    existing = None
    for row in rows:
        if row["team"] == team:
            existing = row
            break

    now = datetime.now(timezone.utc).isoformat()

    if existing:
        old_score = float(existing["accuracy"])
        if accuracy > old_score:
            existing["accuracy"] = accuracy
            existing["submitted_at"] = now
    else:
        rows.append({
            "team": team,
            "accuracy": accuracy,
            "submitted_at": now,
        })

    save_leaderboard(rows)
    return rows


class LeaderboardHandler(SimpleHTTPRequestHandler):
    """Gère les requêtes HTTP pour le leaderboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/upload":
            self.handle_upload()
        else:
            self.send_error(404, "Not Found")

    def handle_upload(self):
        global TEST_LABELS

        if TEST_LABELS is None:
            self.send_json(400, {"error": "Labels de test non charges. Relancez le serveur avec --labels."})
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if "multipart/form-data" not in content_type:
            self.send_json(400, {"error": "Le Content-Type doit etre multipart/form-data"})
            return

        filename = self.headers.get("X-File-Name", "").lower()
        is_encrypted = filename.endswith(".enc")

        try:
            body = self.rfile.read(content_length)
            team_name, file_data, original_filename = self.parse_multipart(body, content_type)
            if original_filename.lower().endswith(".enc"):
                is_encrypted = True
        except Exception as exc:
            self.send_json(400, {"error": f"Erreur de parsing: {exc}"})
            return

        if not team_name:
            self.send_json(400, {"error": "Le nom d'equipe est requis"})
            return

        if not file_data:
            self.send_json(400, {"error": "Le fichier de predictions est requis"})
            return

        import tempfile
        try:
            if is_encrypted:
                # Écrire le fichier chiffré dans un temp, puis le déchiffrer
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name
                with tempfile.NamedTemporaryFile(suffix=".enc", delete=False) as tmp_in:
                    tmp_in_path = tmp_in.name
                    tmp_in.write(file_data)
                
                try:
                    # Importer decrypt_file depuis le module du projet
                    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
                    from encryption.decrypt import decrypt_file
                    decrypt_file(tmp_in_path, tmp_out_path)
                    
                    with open(tmp_out_path, "rb") as f:
                        file_data = f.read()
                except Exception as e:
                    raise ValueError(f"Échec du déchiffrement: {e}. Vérifiez la variable d'environnement PRIVATE_KEY.")
                finally:
                    os.unlink(tmp_in_path)
                    os.unlink(tmp_out_path)

            preds = load_predictions_from_bytes(file_data)
            accuracy = compute_accuracy(preds, TEST_LABELS)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self.send_json(500, {"error": f"Erreur d'evaluation: {exc}"})
            return

        # Sauvegarder le fichier dans submissions/ (toujours en clair ou ce qui a été soumis)
        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        ext = ".enc" if is_encrypted else ".txt"
        submission_path = os.path.join(SUBMISSIONS_DIR, f"{team_name}{ext}")
        with open(submission_path, "wb") as f:
            f.write(file_data if not is_encrypted else body)

        # Mettre à jour le leaderboard
        rows = add_to_leaderboard(team_name, accuracy)

        self.send_json(200, {
            "success": True,
            "team": team_name,
            "accuracy": accuracy,
            "accuracy_pct": f"{accuracy * 100:.2f}%",
            "total_participants": len(rows),
            "rank": next(
                (i + 1 for i, r in enumerate(rows) if r["team"] == team_name),
                len(rows),
            ),
        })
        print(f"  >> {team_name}: accuracy = {accuracy:.6f} ({accuracy * 100:.2f}%)")

    def parse_multipart(self, body: bytes, content_type: str) -> tuple:
        """Parse simplifié du multipart/form-data."""
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary="):]
                break

        if not boundary:
            raise ValueError("Boundary non trouvé dans le Content-Type")

        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)

        team_name = ""
        file_data = b""
        filename = ""

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue

            header = part[:header_end].decode("utf-8", errors="replace")
            data = part[header_end + 4:]
            if data.endswith(b"\r\n"):
                data = data[:-2]
            if data.endswith(b"--\r\n"):
                data = data[:-4]
            if data.endswith(b"--"):
                data = data[:-2]

            if 'name="team"' in header:
                team_name = data.decode("utf-8", errors="replace").strip()
            elif 'name="predictions"' in header:
                file_data = data
                # Extraire le nom de fichier
                for h_part in header.split(";"):
                    h_part = h_part.strip()
                    if h_part.startswith('filename='):
                        filename = h_part[len('filename='):].strip('"\'')

        return team_name, file_data, filename

    def send_json(self, code: int, data: dict):
        response = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Serveur du leaderboard avec evaluation")
    parser.add_argument(
        "--labels",
        default=os.environ.get("TEST_LABELS_PATH", os.path.join(
            os.path.dirname(SCRIPT_DIR), ".tmp", "test_labels.npy"
        )),
        help="Chemin vers test_labels.npy",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port du serveur (default: 8000)")
    args = parser.parse_args()

    global TEST_LABELS
    if os.path.exists(args.labels):
        TEST_LABELS = np.load(args.labels)
        print(f"Labels de test charges: {len(TEST_LABELS)} exemples")
    else:
        print(f"Labels non trouves: {args.labels}")
        print("   L'upload sera desactive. Relancez avec --labels chemin/vers/test_labels.npy")

    server = HTTPServer(("0.0.0.0", args.port), LeaderboardHandler)
    print(f"\nLeaderboard disponible sur: http://localhost:{args.port}")
    print(f"Upload API: http://localhost:{args.port}/api/upload")
    print(f"Appuyez sur Ctrl+C pour arreter\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServeur arrete")
        server.server_close()


if __name__ == "__main__":
    main()
