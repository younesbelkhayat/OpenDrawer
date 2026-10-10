# Directeur — étape `pages` (livre)

## Rôle

Produire chaque page en exécutant **intégralement** la pipeline
`coloring-page` pour chaque photo, dans son propre projet (avec ses propres
checkpoints et gates), avec les mêmes réglages pour tout le livre.

## Règles

- Une page = un projet `coloring-page` (`extra={"book_id": ..., "page_index": n}`
  dans `init_project`). Suivez ses directeurs d'étape.
- Les gates de chaque page restent des gates : présentez les cartoons d'une
  page, attendez la réponse. Pour aller plus vite, proposez de regrouper la
  validation des cartoons de plusieurs pages en un seul message : c'est à
  l'utilisateur de l'accepter (consignez-le, `approval_policy`).
- Mettez à jour le statut de chaque page dans `book_manifest`
  (`pending` → `awaiting_human` → `completed` / `failed`) avec `page_png`
  et `page_svg`.
- Une page en échec n'arrête pas le livre : notez la cause dans `note` et
  continuez ; proposez de la refaire à la fin.

## Gate

Quand toutes les pages sont terminées (ou ignorées d'un commun accord),
checkpoint `awaiting_human` avec le manifeste, présentez la liste, finissez
le tour.
