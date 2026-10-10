# Protocole de checkpoint — meta-skill

## Quand l'utiliser

À la fin de chaque étape (après la relecture), et au début de chaque étape
pour signaler qu'elle est en cours. Les checkpoints sont les points de
sauvegarde : reprise après erreur, validation humaine, historique.

Le CLI (`python -m opendrawer ...`) écrit exactement les mêmes checkpoints :
un projet lancé par le CLI peut être repris par l'agent et inversement.

## 1. Lire la politique du manifeste

```yaml
- name: cartoon
  checkpoint_required: true
  human_approval_default: true   # gate humaine
  produces: [cartoon_set]        # artefacts obligatoires
```

| `human_approval_default` | Action |
|---|---|
| `true` | checkpoint `awaiting_human` → présenter → **finir le tour** → `completed` après accord |
| `false` | checkpoint `completed` puis étape suivante |

Le manifeste fait foi. `lib/checkpoint.py` refuse d'écrire `completed` sur une
étape à gate sans `human_approved=True` (GATE VIOLATION), refuse une étape
dont les prédécesseurs ne sont pas terminés (PREREQUISITE VIOLATION), et
refuse un checkpoint `completed`/`awaiting_human` auquel il manque un
artefact listé dans `produces`.

## 2. Initialiser le projet (une fois)

```python
from lib.checkpoint import init_project
init_project("lea-plage", title="Léa à la plage", pipeline_type="coloring-page",
             style="livre-jeunesse",
             extra={"settings": {"difficulty": "medium", "commercial_safe": False}})
```

Copiez ensuite la photo dans `projects/<id>/input/` et lancez le tableau :
`python -m backlot open <id>` (non bloquant s'il échoue).

Arborescence d'un projet :

```
projects/<id>/
  project.json            marqueur (pipeline, style, réglages)
  input/photo.jpg
  assets/intake/          masque, détourage, masques par sujet
  assets/cartoon/         candidate_1.png …
  assets/line_art/        traits bruts, aplats, masque du sujet
  assets/regions/         labels.png, lines.png, coloring_work.png, model_work.png
  output/                 page.png/svg, coloriage.png/svg, modele.png/svg
  artifacts/*.json        un fichier par artefact
  checkpoint_<étape>.json + history/
  events.jsonl            écrit automatiquement par les outils
```

## 3. Écrire les checkpoints

```python
from lib.checkpoint import PROJECTS_DIR, write_checkpoint
write_checkpoint(PROJECTS_DIR, "lea-plage", "intake", "in_progress", {}, pipeline_type="coloring-page")
# … travail …
write_checkpoint(PROJECTS_DIR, "lea-plage", "intake", "completed",
                 {"photo_analysis": photo_analysis}, pipeline_type="coloring-page")
```

Enregistrez aussi chaque artefact dans `artifacts/<nom>.json` (le tableau et
les étapes suivantes les lisent). Un checkpoint remplacé est archivé dans
`history/` automatiquement.

## 4. Validation humaine

1. Écrire le checkpoint avec `status="awaiting_human"` et les artefacts.
2. Présenter un résumé court : chemins des images à regarder (ou le tableau
   Backlot), votre propre avis issu de la relecture, les choix possibles.
3. **Finir le tour.** Continuer dans la même réponse est une violation.
4. Réponse de l'utilisateur :
   - accord → réécrire `completed` avec `human_approved=True` (et, pour le
     cartoon, `selected_id` = candidat choisi) ;
   - changement demandé → refaire l'étape avec ses retours ;
   - abandon → s'arrêter.
5. Une validation vaut pour une seule gate. Si l'utilisateur autorise
   explicitement tout le parcours (« fais tout sans me demander »),
   consignez-le dans le `decision_log` (`category: "approval_policy"`).

## 5. Échecs et reprise

- Outil en échec → checkpoint `failed` avec `error`, puis appliquer la règle
  « Escalader les blocages » d'`AGENT_GUIDE.md` (aucune substitution sans accord).
- Reprise : `get_next_stage(PROJECTS_DIR, id)` donne l'étape suivante ; une
  étape `awaiting_human` se reprend en re-présentant ses artefacts.

## 6. decision_log

Chaque choix significatif (backend, famille de modèle, mode de ressemblance,
difficulté, candidat retenu, relance avec d'autres paramètres) va dans
`artifacts["decision_log"]` d'un checkpoint ; il est fusionné dans
`projects/<id>/decision_log.json`. Un choix modifié en cours de route = une
nouvelle entrée avec la même paire (`category`, `subject`).
