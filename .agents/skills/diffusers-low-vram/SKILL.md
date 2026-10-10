---
name: diffusers-low-vram
description: Faire tourner SDXL / SD1.5 + ControlNet + IP-Adapter avec diffusers sur un petit GPU (6 Go) — stratégies de déchargement, fp16, OOM, temps attendus. À lire avant diffusers_cartoonize.
---

# diffusers sur petit GPU

## Stratégies mémoire (lib/hardware.py)

| Stratégie | Principe | SDXL sur 6 Go | SD1.5 sur 6 Go |
|---|---|---|---|
| `full_gpu` | tout sur la carte | impossible (~12 Go) | OK, ~10-15 s/image |
| `model_offload` | un sous-modèle à la fois sur le GPU (`enable_model_cpu_offload`) | trop juste (~7 Go) | OK |
| `sequential_offload` | couche par couche (`enable_sequential_cpu_offload`) | OK, 3-6 min/image | OK mais inutile |

`pick_memory_strategy(family)` choisit selon la VRAM ; en cas d'OOM, l'outil
descend l'échelle tout seul (jamais jusqu'au CPU, qui prendrait des heures).
Forcer : `OPENDRAWER_MEMORY_STRATEGY=model_offload`.

Autres réglages appliqués : poids fp16 (`variant="fp16"`), VAE SDXL
`madebyollin/sdxl-vae-fp16-fix` (évite les images noires en fp16), tuilage
du VAE. Prévoir ~16 Go de RAM système pour SDXL avec déchargement.

## Ressemblance

- `plus_face` : IP-Adapter Plus Face (encodeur ViT-H chargé séparément puis
  `load_ip_adapter(..., image_encoder_folder=None)`), image = recadrage
  carré du visage ×1,8. Licence Apache-2.0.
- `faceid` : embeddings InsightFace `buffalo_l` passés via
  `ip_adapter_image_embeds` (forme `(2, 1, 1, 512)` : négatif zéro + positif).
  Meilleure ressemblance, **non commercial**, nécessite `insightface` et
  `onnxruntime-gpu`.
- Sujet sans visage (animal) : échelle IP-Adapter mise à 0 pour ce sujet.

## Plusieurs sujets

Une génération par sujet (recadrage + masque), puis recomposition sur fond
blanc aux positions d'origine : chacun garde son visage.

## Pannes fréquentes

| Symptôme | Cause / remède |
|---|---|
| `ModelNotDownloaded` | `python -m opendrawer models download sdxl` |
| image noire | VAE fp16 manquant → télécharger `sdxl_vae_fp16_fix` |
| OOM même en séquentiel | fermer le navigateur/jeux utilisant le GPU, ou `--family sd15` (demander l'accord) |
| très lent | normal en séquentiel ; annoncer la durée avant de lancer |
