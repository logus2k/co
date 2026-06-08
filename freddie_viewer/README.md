# Freddie 3D Viewer — containerized

Static-site container serving the trained Gaussian-Splatting model of Freddie
via Three.js + `@mkkellogg/gaussian-splats-3d`. Mirrors the Caddy-based
container pattern of `~/env/assets/web_site/` so it slots into the same
reverse-proxy / network setup.

## What gets baked into the image

| Source path | Destination in container | Notes |
|---|---|---|
| `index.html` | `/srv/index.html` | the viewer page |
| `concert_stage.png` | `/srv/concert_stage.png` | toggleable background |
| `src/splats/freddie_gs.ply` | `/srv/src/splats/freddie_gs.ply` | ~13 MB — written by §12.8 of the notebook |
| `node_modules/three/build/` | `/srv/node_modules/three/build/` | 636 KB (the addons subtree is unused and excluded) |
| `node_modules/@mkkellogg/gaussian-splats-3d/build/` | `/srv/node_modules/@mkkellogg/gaussian-splats-3d/build/` | 3.5 MB |
| `freddie_viewer/Caddyfile` | `/etc/caddy/Caddyfile` | listens on :6601 |

Final image is ~70 MB on top of `caddy:2-alpine`.

## Build + run

```bash
cd ~/env/iscte/co/freddie_viewer
docker compose up --build -d
```

Then visit **http://localhost:6601/**.

To stop:

```bash
docker compose down
```

To rebuild after changing `index.html` or re-exporting the PLY:

```bash
docker compose up --build -d
```

## Notes

- The `docker-compose.yml`'s `build.context` is `..` (the project root), so the
  Dockerfile can `COPY` from `node_modules/`, `src/splats/`, and the project
  root without duplicating the assets into this folder. The `.dockerignore` at
  the project root excludes everything else (Python venv, COLMAP data,
  notebook outputs, etc.) so the build context stays small.
- The container joins the existing `logus2k_network` so you can wire it
  behind the same reverse proxy that serves the main site. Remove the
  `networks:` block from `docker-compose.yml` if that network is not present
  on the host.
- The PLY file is sourced from `src/splats/`, which is also the path the
  notebook's `addSplatScene('./src/splats/freddie_gs.ply')` call uses, so the
  same `index.html` works both locally with `npx serve` and inside the
  container without modification.
