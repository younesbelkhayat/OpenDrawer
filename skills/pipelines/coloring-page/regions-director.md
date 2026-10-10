# Directeur — étape `regions` (zones)

## Rôle

Transformer les traits en **zones fermées adaptées à l'âge**, avec la
couleur attendue de chaque zone, et redessiner des traits propres
d'épaisseur uniforme. Produit `region_map`.

## Appel

`region_segment` :

```json
{"line_art": {...}, "palette": "<photo_analysis.palette>",
 "color_source": "photo", "output_dir": "projects/<id>/assets/regions"}
```

Ce que fait l'outil :

- ferme les petits trous des traits, étiquette les zones, puis attribue
  chaque pixel de trait à la zone la plus proche (partition sans trou) ;
- fusionne les zones trop petites ou trop fines pour un crayon avec la
  voisine qui partage la plus longue frontière ;
- **noircit** les petites zones sombres, compactes et entourées par une
  seule zone (pupilles, narines), comme dans les vrais livres ;
- limite le nombre de zones à la plage de la difficulté (fusion des
  voisines de couleurs proches, les petites d'abord) ;
- garde les traits ouverts dessinés (sourire, plis) et les petits points
  d'encre ;
- colore chaque zone avec la couleur du cartoon ramenée à la couleur la plus
  proche de la palette **de la photo** (ΔE ≤ 22).

## Relecture

Ouvrez `regions_preview.png` (zones en couleurs aléatoires), `coloring_work.png`
et `model_work.png`. Puis lancez `coloring_qa` `{region_map}`.

| Problème | Correctif |
|---|---|
| trop de zones pour l'âge | `max_regions` plus bas ou difficulté plus facile |
| détails importants perdus (yeux, bouche) | `min_region_frac` plus petit, ou difficulté supérieure |
| zone ouverte (fuite vers le fond) | autre candidat cartoon, ou `neural: on` à l'étape line_art |
| couleur fausse dans le modèle | `color_source: cartoon` si la palette photo est trompeuse (contre-jour) |

Consignez tout réglage dans le `decision_log` (`parameter_tuning`). Pas de gate.
