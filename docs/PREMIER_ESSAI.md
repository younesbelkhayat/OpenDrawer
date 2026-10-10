# Premier essai sur votre PC (Linux + NVIDIA)

Ce guide mène d'un dépôt fraîchement cloné à une première page de coloriage
générée par diffusion, puis à un rapport à partager en cas de souci.

## 0. Prérequis

- Pilote NVIDIA récent : `nvidia-smi` doit afficher la carte et la VRAM.
- Python 3.10 ou plus récent, ~20 Go de disque libre, 16 Go de RAM conseillés
  pour SDXL sur 6 Go de VRAM.

## 1. Installer

```bash
git clone <votre dépôt> OpenDrawer && cd OpenDrawer
git checkout claude/serene-bell-p6vb4j      # tant que la branche n'est pas fusionnée
make setup                                    # crée .venv + dépendances CPU
source .venv/bin/activate

# torch CUDA : choisissez la roue qui correspond à votre pilote (cu121, cu124…)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
make install-gpu                              # diffusers, transformers, rembg[gpu]…
python -m opendrawer demo                     # vérifie la chaîne sans GPU (quelques secondes)
```

## 2. Télécharger les modèles (une seule fois, sans compte)

```bash
python -m opendrawer models list              # licences et tailles
python -m opendrawer models download sd15 analysis    # ~6 Go, le plus rapide pour commencer
python -m opendrawer models download sdxl             # ~13 Go, la meilleure qualité
# optionnel, NON commercial : meilleure ressemblance
pip install insightface onnxruntime-gpu && python -m opendrawer models download faceid
```

Les poids vont dans `models/` (ou `OPENDRAWER_MODELS_DIR`). rembg et
insightface téléchargent aussi leurs petits modèles au premier usage.

## 3. Autodiagnostic

```bash
python -m opendrawer selftest --family sd15     # 1 génération courte (12 étapes)
python -m opendrawer selftest --family sdxl     # plus long sur 6 Go (déchargement séquentiel)
```

Chaque ligne ✔/✗ correspond à une brique : versions, GPU, modèles, détourage,
détection, puis une page complète. Le rapport est écrit dans
`projects/_selftest_report.json` : **c'est ce fichier à me renvoyer** si une
ligne est en ✗ (il ne contient ni photo ni donnée personnelle, seulement des
versions, des temps et des messages d'erreur).

## 4. Première vraie page

```bash
python -m opendrawer page ma_photo.jpg --difficulty facile --family sd15
python -m backlot open        # dans un autre terminal, pour suivre en direct
```

Le terminal propose les cartoons : tapez le numéro à garder (Entrée = recommandé).
Puis il montre la page et le contrôle qualité : `o` pour valider.

Astuces si le résultat déçoit :

| Problème | Essai |
|---|---|
| pas assez « cartoon » | `--family sdxl`, ou relancer (nouvelle graine) |
| visage peu ressemblant | `--identity faceid` (non commercial) ou `--describe s1="a girl with short curly brown hair and glasses"` |
| trop de zones pour un petit | `--difficulty facile` |
| mémoire GPU insuffisante | fermer le navigateur/jeux, `--family sd15`, ou `OPENDRAWER_MEMORY_STRATEGY=sequential_offload` |

## 5. Avec un agent

Ouvrez le dossier dans Claude Code (ou Codex, Cursor) et tapez
`/coloriage ma_photo.jpg facile` : l'agent suit les pipelines, regarde les
images et vous demande de valider le cartoon puis la page.
