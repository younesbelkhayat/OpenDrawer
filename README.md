# OpenDrawer

**Une photo → une page de coloriage pour enfant.** OpenDrawer transforme la
photo d'une personne (ou de plusieurs, ou d'un animal) en illustration style
*livre jeunesse*, puis en dessin au trait à zones fermées, accompagné d'un
**modèle colorié aux vraies couleurs de la photo** pour que l'enfant sache
quelle couleur mettre où.

- 🖍️ Page A4 300 dpi : grand dessin à colorier + vignette du modèle colorié
- 🧩 SVG à zones remplissables (`data-color` = couleur attendue de chaque zone)
- 🎚️ Trois difficultés : facile (3-5 ans), moyen (6-8 ans), difficile (9-12 ans)
- 📚 Un dossier de photos → un livre de coloriage multi-pages
- 🔒 100 % local : aucune photo ne quitte votre machine
- 🤖 Pilotable par un agent IA (Claude Code, Codex, Cursor…) ou en ligne de commande

## Installation

```bash
make setup          # dépendances de base (CPU)
make install-gpu    # torch CUDA + diffusers (GPU NVIDIA, 6 Go suffisent)
make models         # SDXL + ControlNet + IP-Adapter, ~12 Go, une seule fois, sans compte
python -m opendrawer doctor
```

Pas de GPU ? `python -m opendrawer demo` produit une page de démonstration
en quelques secondes avec le backend OpenCV (aucun modèle).

## Utilisation

```bash
# Une page
python -m opendrawer page photo.jpg --difficulty facile

# Un livre (photos triées par nom : 01_lea.jpg, 02_tom.jpg…)
python -m opendrawer book mes_photos/ --difficulty moyen

# Reprendre un projet en attente de validation
python -m opendrawer resume lea

# Suivre en direct dans le navigateur
python -m backlot open
```

À chaque étape à valider, le terminal propose les cartoons (choix du
meilleur) puis la page finale. `--yes` valide automatiquement le candidat
recommandé.

Options utiles : `--backend diffusers|comfyui|opencv`, `--family sdxl|sd15`,
`--identity faceid|plus_face|none`, `--candidates 4`, `--seed 42`,
`--describe s1="a girl with curly red hair and glasses"`,
`--commercial-safe`.

## Sorties

```
projects/<projet>/output/
  page.png / page.svg          page A4 combinée
  coloriage.png / .svg         le dessin à colorier seul
  modele.png / .svg            le modèle colorié
```

## Comment ça marche

| Étape | Ce qui se passe |
|---|---|
| photo | détection des personnes/animaux et des visages, détourage sur fond blanc, palette des vraies couleurs |
| cartoon | img2img + ControlNet + IP-Adapter (SDXL ou SD1.5) : style livre jeunesse, ressemblance conservée — **validation** |
| traits | contours d'encre, frontières d'aplats, silhouette |
| zones | zones fermées adaptées à l'âge, petits détails noircis, couleurs de la photo |
| page | SVG, mise en page A4, contrôle qualité — **validation** |

Détails : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Avec un agent IA

Ouvrez le dépôt dans votre agent et demandez par exemple « fais un
coloriage facile de cette photo ». L'agent suit [`AGENT_GUIDE.md`](AGENT_GUIDE.md) :
pipelines YAML, directeurs d'étape, relecture des images, validations.

## Licences

Code sous [AGPL-3.0](LICENSE). Chaque modèle a sa propre licence, indiquée
dans [`models.yaml`](models.yaml) (`python -m opendrawer models list`).
`--commercial-safe` n'utilise que des modèles à licence permissive vérifiée ;
la ressemblance FaceID (InsightFace) est **non commerciale**. Vérifiez
toujours la fiche d'un modèle avant un usage commercial.

OpenDrawer est dérivé d'[OpenMontage](https://github.com/calesthio/OpenMontage)
(AGPL-3.0), dont il reprend l'architecture.
