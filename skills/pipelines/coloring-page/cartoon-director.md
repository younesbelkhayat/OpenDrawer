# Directeur — étape `cartoon`

## Rôle

Transformer la photo détourée en illustration **livre jeunesse** (formes
rondes, contours noirs nets, aplats) qui reste **ressemblante** et garde les
couleurs de la photo. Produit `cartoon_set` (plusieurs candidats, un retenu).
**Gate humaine.**

Avant d'appeler un outil, lisez `.agents/skills/cartoon-prompting/SKILL.md`,
et `.agents/skills/diffusers-low-vram/SKILL.md` (backend diffusers) ou
`.agents/skills/comfyui/SKILL.md` (backend ComfyUI).

## Choix à annoncer avant de lancer

Annoncez outil, backend, famille (`sdxl` / `sd15`), mode de ressemblance
(`faceid` non commercial, `plus_face`, `none`), nombre de candidats et durée
estimée (`tool.estimate_runtime(inputs)`). Consignez-les dans le
`decision_log` (`backend_selection`, `model_selection`, `identity_mode`).
Ne changez jamais de backend ou de famille après un échec sans accord.

## Appel

`cartoonize` (sélecteur) ou directement le backend :

```json
{"photo_analysis": {...}, "output_dir": "projects/<id>/assets/cartoon",
 "backend": "diffusers", "family": "sdxl", "identity": "auto",
 "candidates": 3, "style": "livre-jeunesse",
 "descriptions": {"s1": "a smiling girl with curly red hair and glasses"}}
```

Les candidats sont alignés sur la photo (même cadrage) : c'est ce qui permet
de reprendre les vraies couleurs aux étapes suivantes. Ne recadrez pas les
images produites.

## Relecture (obligatoire, en regardant chaque candidat)

- ressemblance : coiffure, lunettes, couleur de cheveux, vêtements, accessoires ;
- style : contours nets, aplats, pas de texture/ombrage photo ;
- défauts : doigts/membres en trop, yeux asymétriques, visage fondu ;
- fond blanc, sujet entier.

Le score objectif (`score.total`) n'est qu'un indice. Recommandez le
meilleur candidat selon **votre** regard et expliquez pourquoi.

Ajustements utiles (même backend, même famille) : nouvelle `seed` ;
`strength` +0,05 (plus cartoon) / −0,05 (plus fidèle) ; `ip_adapter_scale`
↑ si le visage n'est pas reconnaissable ; `controlnet_scale` ↑ si la pose
dérive ; description plus précise.

## Gate

Checkpoint `awaiting_human` avec `selected_id` = votre recommandation,
présentez les chemins des candidats (ou le tableau Backlot), finissez le
tour. Après la réponse, réécrivez `completed`, `human_approved=True`,
`selected_id` = choix de l'utilisateur et `selection_reason`.
