# Directeur — étape `assemble` (livre)

## Rôle

Rassembler les pages validées dans `projects/<livre>/output/livre/` :
`page_01.png` / `page_01.svg`… numérotées dans l'ordre, et une planche
`sommaire.png`.

## Appel

`book_assemble` `{book_manifest, output_dir: projects/<livre>/output/livre}`
(par défaut seules les pages `completed` sont incluses).

## Gate

Présentez le dossier et `sommaire.png`, vérifiez qu'aucune page ne manque,
finissez le tour. Après accord : `completed`, `human_approved=True`.
