# Web Pocket Viewer

Static browser demo and embeddable custom element for the cylindrical oblong pocket model.

## Run locally

Serve the repo root over HTTP, then open the demo page:

```powershell
python -m http.server 8000
```

Then browse to `http://localhost:8000/web/`.

## Embed

```html
<script type="module" src="./cylindrical-oblong-pocket-viewer.js"></script>
<cylindrical-oblong-pocket-viewer></cylindrical-oblong-pocket-viewer>
```

The component includes:

- Built-in controls for the main pocket, toolpath, and pattern parameters.
- A 3D cylinder mesh with constant radial pocket depth.
- Optional boundary and spiral toolpath overlays.
- A `pocketchange` event with the current parameter payload in `event.detail`.
