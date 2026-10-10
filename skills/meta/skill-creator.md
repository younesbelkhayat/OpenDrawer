# Créer un skill — meta-skill

## Quand l'utiliser

Pour ajouter un savoir réutilisable : un nouveau style de cartoon, une
astuce de prompt, une étape de pipeline, un nouveau backend.

## Trois couches

| Couche | Emplacement | Contenu |
|---|---|---|
| 1 | `tools/` (registry) | ce qui existe : contrats `BaseTool` |
| 2 | `skills/` | comment OpenDrawer s'en sert (directeurs d'étape, meta-skills) |
| 3 | `.agents/skills/<nom>/SKILL.md` | comment marche la technologie (diffusers, ComfyUI, prompts) |

Un outil pointe vers ses skills de couche 3 via `agent_skills`.

## Recettes

- **Nouveau style** : copier `styles/livre-jeunesse.yaml`, changer `name`,
  prompts et paramètres ; il est validé par `schemas/styles/style.schema.json`
  puis utilisable via `--style <nom>`. L'ajouter à `compatible_styles` des
  manifestes.
- **Nouveau directeur d'étape** : `skills/pipelines/<pipeline>/<étape>-director.md`
  avec : rôle, entrées, outils (dans l'ordre), paramètres à ajuster, critères
  de sortie, pièges connus. Le référencer dans le champ `skill` du manifeste.
- **Nouveau skill de couche 3** : dossier `.agents/skills/<nom>/` avec un
  `SKILL.md` commençant par un front-matter `name` / `description`.

Les tests de contrat (`tests/contracts/`) vérifient que tout skill référencé
existe.
