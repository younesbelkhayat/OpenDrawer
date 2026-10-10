# OpenDrawer — contexte du projet

Source de vérité pour l'architecture et les conventions. Les fichiers propres
à chaque agent (CLAUDE.md, CODEX.md, CURSOR.md, COPILOT.md) pointent ici.

## Identité

OpenDrawer transforme une photo en page de coloriage pour enfant (cartoon
livre jeunesse, zones fermées, modèle colorié aux vraies couleurs), 100 %
local, pilotable par un agent IA ou par une ligne de commande.

## Architecture : pilotée par les instructions

```
l'agent lit le manifeste (YAML) → lit le directeur d'étape (MD)
→ appelle les outils (Python BaseTool) → relit (meta-skill, en regardant les images)
→ checkpoint (lib/checkpoint.py) → validation humaine aux gates
```

Python = outils + persistance. L'orchestration, les choix créatifs et la
relecture vivent dans les instructions. Exception assumée : `opendrawer/runner.py`
enchaîne les mêmes outils pour le CLI, sans logique créative propre.

## Trois couches de connaissance

```
Couche 1 : tools/            ce qui existe (contrats BaseTool, registry)
Couche 2 : skills/           comment OpenDrawer s'en sert (directeurs, meta-skills)
Couche 3 : .agents/skills/   comment marche la technologie (diffusers, ComfyUI, prompts)
```

## Pipelines

| Pipeline | Étapes | Gates |
|---|---|---|
| `coloring-page` | intake → cartoon → line_art → regions → page | cartoon, page |
| `coloring-book` | collect → pages → assemble | pages, assemble |

## Artefacts (schemas/artifacts/)

`photo_analysis`, `cartoon_set`, `line_art`, `region_map`, `coloring_page`,
`qa_report`, `book_manifest`, plus `review` et `decision_log`. Les artefacts
obligatoires d'une étape = son `produces` dans le manifeste.

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `tools/base_tool.py` | contrat des outils + instrumentation Backlot |
| `tools/tool_registry.py` | découverte, menu de préflight |
| `tools/cartoon/_provider.py` | contrat commun des backends de cartoonisation |
| `tools/coloring/region_segment.py` | zones fermées, fusions, remplissages noirs, couleurs |
| `lib/checkpoint.py` | checkpoints, gates, historique |
| `lib/model_registry.py` + `models.yaml` | poids locaux, licences, téléchargement explicite |
| `lib/hardware.py` | GPU, VRAM, stratégie mémoire |
| `lib/difficulty.py` | préréglages facile / moyen / difficile |
| `styles/*.yaml` | styles de cartoon (prompts, paramètres) |
| `opendrawer/cli.py`, `opendrawer/runner.py` | ligne de commande |
| `backlot/` | tableau local en direct |

## Conventions pour un nouvel outil

1. Hériter de `BaseTool`, le placer dans le bon paquet (`analysis`,
   `segmentation`, `cartoon`, `coloring`).
2. Renseigner `name`, `capability`, `provider`, `runtime`, `dependencies`,
   `input_schema`, `required_models` (clés de `models.yaml`), `agent_skills`.
3. `execute()` renvoie un `ToolResult` ; ne jamais télécharger de poids
   (passer par `model_registry.path()`), ne jamais ouvrir de connexion réseau.
4. Préférer sélecteur + fournisseurs (`cartoonize` → backends).
5. Ajouter des tests sans GPU (backend OpenCV, `opendrawer.demo`).

## Conventions de langue

Code, identifiants et commentaires en anglais ; documentation, skills et
messages à l'utilisateur en français.
