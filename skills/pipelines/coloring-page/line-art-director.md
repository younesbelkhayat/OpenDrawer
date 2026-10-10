# Directeur — étape `line_art` (traits)

## Rôle

Extraire du cartoon retenu un masque de traits **généreux** (les zones
seront fermées et nettoyées à l'étape suivante). Produit `line_art`.

## Appel

`lineart_extract` :

```json
{"cartoon_path": "<chemin du candidat retenu>",
 "foreground_mask_path": "<photo_analysis.foreground_mask_path>",
 "output_dir": "projects/<id>/assets/line_art",
 "difficulty": "medium", "neural": "auto", "commercial_safe": false}
```

Indices combinés : contours sombres dessinés (encre, L* < 26), frontières
entre aplats de couleur, silhouette, et, si les poids `lineart_annotators`
sont présents (licence non vérifiée : jamais en commercial-safe), un
annotateur lineart neuronal pour les détails du visage.

Le travail se fait à 2400 px sur le grand côté : la page A4 ne fait que
réduire, donc les traits restent nets.

## Relecture

Ouvrez `line_mask.png` (traits bruts) et `flat.png` (aplats) : silhouette
fermée, visage lisible, pas de grain transformé en traits. Si le cartoon
est trop texturé, mieux vaut revenir à l'étape cartoon (autre candidat) que
forcer ici. Pas de gate humaine.
