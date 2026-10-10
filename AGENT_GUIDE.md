# OpenDrawer — guide de l'agent

Point d'entrée obligatoire. Ce guide est le contrat de l'agent ; l'architecture
et les conventions sont dans [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Ce que fait OpenDrawer

Une photo (une ou plusieurs personnes, des animaux) devient une **page de
coloriage pour enfant** : cartoon style livre jeunesse → dessin au trait à
zones fermées → **modèle colorié aux vraies couleurs de la photo** → page A4
(grand dessin + vignette du modèle), en PNG et en SVG à zones remplissables.
Un dossier de photos peut devenir un **livre** multi-pages.

Tout tourne **en local** (GPU NVIDIA conseillé, 6 Go suffisent) : aucune
photo ne quitte la machine. Les poids des modèles sont téléchargés une seule
fois, sans compte (`python -m opendrawer models download ...`).

## Routage du premier message

| L'utilisateur… | Première action |
|---|---|
| est vague (« que sais-tu faire ? ») | lire `skills/meta/onboarding.md` |
| donne une photo / veut une page | pipeline `coloring-page` (règle zéro) |
| donne un dossier / veut un livre | pipeline `coloring-book` |
| veut reprendre un projet | `get_next_stage()` puis l'étape indiquée |
| demande de régler un résultat | relire l'artefact concerné, puis le directeur de l'étape |

## Règle zéro — toute production passe par une pipeline

1. **Identifier la pipeline** dans `pipeline_defs/` (`coloring-page`, `coloring-book`).
2. **Lire le manifeste** : étapes, outils, gates, `review_focus`.
3. **Préflight** : `python -m opendrawer doctor` ou
   `registry.provider_menu_summary()` ; présenter ce qui est disponible
   (GPU, backends, modèles manquants) et le temps estimé.
4. **Étape par étape** : lire le directeur
   `skills/pipelines/<pipeline>/<étape>-director.md` **avant** d'agir.
5. **Lire les skills de couche 3** (`.agents/skills/`) référencés par les
   outils (`agent_skills`) avant de les appeler.
6. **Relire** (`skills/meta/reviewer.md`) en **regardant les images**, puis
   **checkpointer** (`skills/meta/checkpoint-protocol.md`).

À ne pas faire : écrire des scripts ad hoc qui contournent les outils, sauter
une étape, générer sans avoir lu le directeur, présenter sans avoir regardé.

Le CLI (`python -m opendrawer page|book|resume`) exécute les mêmes outils et
écrit les mêmes checkpoints ; c'est la voie « sans agent ». Un projet
démarré par le CLI peut être repris par l'agent et inversement.

## Contrat de communication des décisions

### Annoncer avant d'exécuter

Avant toute génération longue (cartoonisation surtout), indiquer : l'outil,
le backend (diffusers / comfyui / opencv), la famille (sdxl / sd15), le mode
de ressemblance (faceid / plus_face / none), le nombre de candidats, la durée
estimée, la raison du choix.

### Demander avant un changement majeur

Changer de backend, de famille de modèle, de mode de ressemblance, de style,
de difficulté, passer en `--commercial-safe` ou en sortir, ou abandonner une
étape : **toujours demander**. Les ajustements fins dans un choix approuvé
(graine, `strength`, `max_regions`…) n'exigent pas d'accord mais vont dans
le `decision_log`.

### Escalader les blocages

En cas d'échec : 1) ce qui a été tenté, 2) ce qui a échoué, 3) la nature
(modèle manquant, mémoire GPU, bug d'outil, qualité), 4) les options,
5) votre recommandation. Aucune substitution (ex. repli sur OpenCV) sans accord.

### decision_log

Append-only. Un choix modifié = nouvelle entrée avec la même paire
(`category`, `subject`), l'ancien choix dans `options_considered` avec
`rejected_because`.

## Licences et `--commercial-safe`

Le catalogue `models.yaml` marque chaque modèle `commercial_safe`. En mode
commercial-safe (`OPENDRAWER_COMMERCIAL_SAFE=1` ou option CLI), le registre
refuse les modèles non commerciaux (IP-Adapter FaceID / InsightFace) et ceux
dont la licence n'est pas vérifiée (annotateur lineart). Ne présentez jamais
une licence comme vérifiée si le catalogue ne le dit pas.

## Vie privée

Les photos sont souvent celles d'enfants. Ne les envoyez à aucun service,
ne les copiez pas hors de `projects/`, ne les décrivez pas au-delà de ce qui
sert le prompt. Les tests bloquent tout accès réseau.

## Tableau Backlot

Après `init_project`, lancez `python -m backlot open <projet>` : le tableau
montre en direct chaque étape (photo, cartoons, traits, zones, page, QA,
décisions). Il observe seulement le disque ; s'il échoue, continuez.

## Fichiers de référence

| Sujet | Fichier |
|---|---|
| Architecture, conventions | `PROJECT_CONTEXT.md`, `docs/ARCHITECTURE.md` |
| Index des skills | `skills/INDEX.md` |
| Registre des outils | `tools/tool_registry.py` |
| Manifestes | `pipeline_defs/*.yaml` |
| Schémas d'artefacts | `schemas/artifacts/` |
| Catalogue de modèles | `models.yaml`, `lib/model_registry.py` |
| Difficultés | `lib/difficulty.py` |
| Styles | `styles/*.yaml` |
