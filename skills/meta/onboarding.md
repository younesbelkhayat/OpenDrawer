# Onboarding — meta-skill

## Quand l'utiliser

Premier message vague (« qu'est-ce que tu sais faire ? », « je veux faire un
coloriage »). Si l'utilisateur fournit déjà une photo et une demande claire,
passez directement à la pipeline.

## 1. Découvrir la machine

```bash
python -m opendrawer doctor
```

ou en Python : `registry.provider_menu_summary()` (GPU, VRAM, outils
disponibles, modèles manquants).

## 2. Classer la configuration

| Niveau | Ce qui est disponible | Ce qu'on peut promettre |
|---|---|---|
| **Sans GPU / sans modèle** | `opencv_cartoonize`, GrabCut, Haar | aperçu fidèle à la photo mais peu « cartoon » ; idéal pour tester |
| **GPU + SD1.5** | `diffusers_cartoonize` famille sd15 | cartoon correct, ~30 s à 1 min par image sur 6 Go |
| **GPU + SDXL** | `diffusers_cartoonize` famille sdxl (+ IP-Adapter) | meilleure qualité ; 6 Go → déchargement séquentiel, 3 à 6 min par image |
| **ComfyUI** | `comfyui_cartoonize` | même recette avec vos propres modèles/LoRA |

Modèles manquants → proposer `python -m opendrawer models download sdxl analysis`
(téléchargement unique, sans compte ; ensuite tout est hors ligne).

## 3. Présenter simplement

Expliquer en quelques lignes : une photo (personnes et/ou animaux) devient
une page A4 à colorier, avec une vignette du modèle colorié aux vraies
couleurs de la photo ; PNG pour imprimer, SVG à zones remplissables ; trois
difficultés (facile 3-5 ans, moyen 6-8 ans, difficile 9-12 ans) ; un dossier
de photos peut devenir un livre.

## 4. Questions à poser (QCM courts)

- difficulté (facile / moyen / difficile) ;
- une page ou un livre ;
- usage commercial possible ? (→ `--commercial-safe`, exclut FaceID et les
  modèles à licence non vérifiée) ;
- détails à souligner pour la ressemblance (lunettes, couettes, doudou…).

## 5. Démarrage rapide

```bash
python -m opendrawer demo                         # vérifie l'installation (sans GPU)
python -m opendrawer page photo.jpg --difficulty facile
python -m opendrawer book dossier/ --difficulty moyen
```

Avec l'agent, suivre `pipeline_defs/coloring-page.yaml` étape par étape.
