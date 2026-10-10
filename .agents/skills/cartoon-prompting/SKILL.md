---
name: cartoon-prompting
description: Prompter et régler une cartoonisation img2img + ControlNet pour obtenir un style livre jeunesse ressemblant et facile à transformer en coloriage. À lire avant toute étape cartoon.
---

# Prompts et réglages cartoon

## Ce qu'on veut

Un dessin qui se transforme bien en coloriage : **contours noirs nets**,
**aplats** sans dégradé ni texture, formes simples, fond blanc, sujet
entier — tout en restant reconnaissable.

## Prompt

Le style (`styles/livre-jeunesse.yaml`) fournit le gabarit ; vous fournissez
`{subject}` via `descriptions` (en anglais) :

- traits distinctifs d'abord : coiffure (« curly shoulder-length red hair »,
  « short spiky blond hair »), lunettes, barbe, couettes, appareil dentaire ;
- vêtements visibles : « yellow raincoat », « striped t-shirt » ;
- animaux : race et pelage (« a fluffy white bichon dog ») ;
- l'âge apparent (« a little boy », « a grandmother »).

Évitez les détails impossibles à colorier (« intricate lace », « freckles
everywhere ») pour les difficultés facile/moyen.

## Paramètres (img2img + ControlNet Canny)

| Paramètre | Effet | Défaut SDXL | Plage utile |
|---|---|---|---|
| `strength` | 0 = photo, 1 = ignore la photo | 0,72 | 0,6 – 0,8 |
| `controlnet_scale` | fidélité aux contours de la photo (pose, silhouette) | 0,55 | 0,4 – 0,8 |
| `ip_adapter_scale` | fidélité au visage | 0,6 | 0,4 – 0,8 |
| `guidance_scale` | adhérence au prompt | 6 | 5 – 8 |
| `steps` | qualité / temps | 30 | 25 – 40 |

Règle de pouce : pas assez cartoon → `strength` ↑ ; visage méconnaissable →
`ip_adapter_scale` ↑ puis `strength` ↓ ; pose ou bras qui dérivent →
`controlnet_scale` ↑ ; trop de détails photo → `controlnet_scale` ↓.

Changez un paramètre à la fois et gardez la même `seed` pour comparer.
Consignez chaque réglage dans le `decision_log` (`parameter_tuning`).
