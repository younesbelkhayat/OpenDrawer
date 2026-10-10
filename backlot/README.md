# Backlot — le tableau en direct

Tableau local en lecture seule qui montre un projet OpenDrawer en train de
se faire : étapes qui s'allument, photo et détourage, propositions de
cartoon (avec le candidat retenu), traits, zones, page finale, contrôle
qualité, décisions et activité des outils. Pour un livre : chaque page avec
son statut et la planche-sommaire.

```bash
python -m backlot open <projet>   # démarre le serveur si besoin + ouvre le navigateur
python -m backlot open            # tous les projets
python -m backlot serve --port 4750
```

## Sources

| Élément | Fichier |
|---|---|
| identité, ordre des étapes | `project.json` + `pipeline_defs/<type>.yaml` |
| état des étapes, gates, versions | `checkpoint_<étape>.json` + `history/` |
| images par étape | artefacts `photo_analysis`, `cartoon_set`, `line_art`, `region_map`, `coloring_page` |
| contrôle qualité | `artifacts/qa_report.json` |
| pages d'un livre | `artifacts/book_manifest.json` |
| activité | `events.jsonl` (instrumentation de `BaseTool`) |

Aucune police web, aucune ressource distante : le tableau fonctionne hors ligne.
