---
name: comfyui
description: Utiliser un serveur ComfyUI local comme backend de cartoonisation (comfyui_cartoonize) — configuration, noms de modèles, nœuds requis, IP-Adapter optionnel.
---

# ComfyUI avec OpenDrawer

## Contrat serveur

- Serveur local sur `http://localhost:8188` (ou `COMFYUI_SERVER_URL` /
  `COMFYUI_IMAGE_SERVER_URL`). Santé : `GET /system_stats`.
- Soumission `POST /prompt`, résultat `GET /history/{id}`, fichier
  `GET /view`, envoi d'image `POST /upload/image`.
- Un délai dépassé est récupérable : l'erreur porte `prompt_id`.

## Workflow utilisé

Construit en Python (`tools/cartoon/comfyui_cartoonize.py: build_workflow`)
avec des nœuds **de base** : CheckpointLoaderSimple, CLIPTextEncode ×2,
LoadImage, ImageScale, Canny, ControlNetLoader, ControlNetApplyAdvanced,
VAEEncode, KSampler (`denoise` = strength), VAEDecode, SaveImage.

Si les nœuds personnalisés **ComfyUI_IPAdapter_plus** sont installés
(`IPAdapterUnifiedLoader`, `IPAdapter`), la ressemblance du visage est
ajoutée automatiquement (preset « PLUS FACE (portraits) »). FaceID n'est pas
pris en charge par ce backend.

## Noms de modèles

Ce sont les noms de fichiers dans les dossiers de ComfyUI :

- `COMFYUI_CARTOON_CHECKPOINT` (défaut `sd_xl_base_1.0.safetensors`) — vous
  pouvez y mettre un checkpoint cartoon personnel ;
- `COMFYUI_CARTOON_CONTROLNET` (défaut `controlnet-canny-sdxl-1.0-small.safetensors`).

Pour SD1.5 (`family: sd15`) : `v1-5-pruned-emaonly.safetensors` et
`control_v11p_sd15_canny.safetensors` par défaut.

## Petits GPU

Lancer ComfyUI avec `--lowvram` (6 Go) ; SDXL reste plus lent que SD1.5.
