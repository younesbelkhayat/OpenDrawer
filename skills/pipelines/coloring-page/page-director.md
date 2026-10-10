# Directeur — étape `page`

## Rôle

Produire les livrables : SVG à zones remplissables, page A4 combinée (grand
dessin + vignette du modèle colorié), PNG haute résolution, rapport QA.
Produit `coloring_page` et `qa_report`. **Gate humaine.**

## Outils, dans l'ordre

1. `vectorize_svg` `{region_map, output_dir: projects/<id>/output}` →
   `coloriage.svg` (zones blanches) et `modele.svg` (zones colorées). Chaque
   zone : `<path class="zone" data-color="#…" data-color-name="…">`.
2. `page_layout` `{region_map, coloring_svg, model_svg, output_dir}` →
   `page.png` (A4 300 dpi), `page.svg`, `coloriage.png`, `modele.png`. Le
   dessin est agrandi au maximum sans que la vignette (coin bas-droit) ne
   touche ses traits.
3. `coloring_qa` `{region_map, coloring_page, output_path: artifacts/qa_report.json}`.

## Relecture

Ouvrez `page.png` en réduit puis un recadrage à 100 % (visage) : traits
nets, pas de pixellisation, vignette lisible, aucune zone microscopique.
Un `fail` QA bloque la présentation : corrigez à l'étape concernée.

## Gate

Checkpoint `awaiting_human` ; présentez `page.png`, les SVG, le statut QA
et vos remarques ; finissez le tour. Après accord : `completed`,
`human_approved=True`. Si c'est une page d'un livre, la pipeline
`coloring-book` reprend la main.
