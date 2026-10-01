# Antumbra brand assets

The Antumbra mark is an annulus: the ring of light you see around the moon
from inside the antumbra during an annular eclipse. The ring sits off
center, heavier on one side, and the periwinkle bead marks the point where
the light breaks through.

## Files

| File | Size | Use it for |
| --- | --- | --- |
| `banner.png` | 2560x1280 | The header image at the top of the README. |
| `social-preview.png` | 1280x640 | The GitHub social preview (see below). Also works for link cards and slides. |
| `icon.svg` | 512x512 | The app icon: the mark on a dark rounded tile. Use it for avatars, package registries, and documentation sites. |
| `icon-512.png` | 512x512 | The same icon as a PNG, for places that do not accept SVG. |
| `favicon-32.png` | 32x32 | A browser tab icon. |
| `mark-dark.svg` | 512x512 | The bare mark for dark backgrounds. The background is transparent. |
| `mark-light.svg` | 512x512 | The bare mark for light backgrounds. The background is transparent. |
| `lockup-dark.png` | 1384x352 | The mark with the `antumbra` wordmark and tagline on a dark tile. |
| `lockup-light.png` | 1384x352 | The mark with the `antumbra` wordmark and tagline on a light tile. |

To show the right bare mark for the reader's GitHub theme:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mark-dark.svg">
  <img src="assets/mark-light.svg" alt="Antumbra" width="96">
</picture>
```

## Colors

| Name | Hex | Role |
| --- | --- | --- |
| Night | `#12121B` | Dark ground. |
| Pearl | `#EFE7D8` | The ring and the wordmark on dark backgrounds. |
| Periwinkle | `#8A84CC` | The bead on dark backgrounds, and the accent color. |
| Ink | `#1B1A2E` | The ring and the wordmark on light backgrounds. |
| Periwinkle (light) | `#6A63B8` | The bead on light backgrounds. |
| Mist | `#EEECF5` | Light ground. |

The wordmark is set in Source Sans Pro, weight 300, with wide letter
spacing: lowercase in the lockups and capitals on the banner. The lockups
carry the tagline "Memory that gets better." and the banner carries
"Private Agent Substrate", both in Source Sans Pro, weight 400.

## Social preview

GitHub does not read the social preview from the repository. To set it,
open the repository's Settings, and under General > Social preview, upload
`social-preview.png`.

## Source

The artwork is drawn in Penpot, in the Brands project, file "Repository
Brands". The mark is the "antumbra mark" component with Concept "A1
Annulus" and Ink "Colour". The banner is the antumbra README banner board on
the Family page. Export from there if you need another size or format.
