# Index des skills OpenDrawer

## Couche 2 — comment OpenDrawer travaille

### Meta-skills

| Skill | Quand |
|---|---|
| `meta/onboarding.md` | premier contact, demande vague |
| `meta/checkpoint-protocol.md` | début/fin de chaque étape, gates humaines |
| `meta/reviewer.md` | relecture après chaque étape (regarder les images) |
| `meta/skill-creator.md` | ajouter un style, un skill, un directeur |

### Pipeline `coloring-page`

| Étape | Directeur | Gate |
|---|---|---|
| intake | `pipelines/coloring-page/intake-director.md` | non |
| cartoon | `pipelines/coloring-page/cartoon-director.md` | **oui** |
| line_art | `pipelines/coloring-page/line-art-director.md` | non |
| regions | `pipelines/coloring-page/regions-director.md` | non |
| page | `pipelines/coloring-page/page-director.md` | **oui** |

### Pipeline `coloring-book`

| Étape | Directeur | Gate |
|---|---|---|
| collect | `pipelines/coloring-book/collect-director.md` | non |
| pages | `pipelines/coloring-book/pages-director.md` | **oui** |
| assemble | `pipelines/coloring-book/assemble-director.md` | **oui** |

## Couche 3 — comment marche la technologie (`.agents/skills/`)

| Skill | Outils |
|---|---|
| `diffusers-low-vram` | `diffusers_cartoonize` |
| `cartoon-prompting` | `cartoonize`, `diffusers_cartoonize`, `comfyui_cartoonize` |
| `comfyui` | `comfyui_cartoonize` |

## Outils (couche 1, `tools/`)

| Capacité | Outils |
|---|---|
| détection | `subject_detect` |
| segmentation | `bg_remove` |
| couleurs | `palette_extract` |
| cartoon | `cartoonize` (sélecteur) → `diffusers_cartoonize`, `comfyui_cartoonize`, `opencv_cartoonize` |
| traits | `lineart_extract` |
| zones | `region_segment` |
| vectorisation | `vectorize_svg` |
| mise en page | `page_layout` |
| qualité | `coloring_qa` |
| livre | `book_assemble` |
