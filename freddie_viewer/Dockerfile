FROM caddy:2-alpine

# Caddy config bundled next to this Dockerfile
COPY freddie_viewer/Caddyfile /etc/caddy/Caddyfile

# Web entrypoint + background image
COPY index.html      /srv/
COPY concert_stage.png /srv/

# Trained Gaussian-Splatting model (~13 MB, written by §12.8 of the notebook)
COPY src/splats/freddie_gs.ply /srv/src/splats/

# Only the JS module trees referenced by the import map.
# - three: just the build/ folder (636 KB); the examples/jsm addons are unused.
# - @mkkellogg/gaussian-splats-3d: the build/ folder (3.5 MB).
COPY node_modules/three/build /srv/node_modules/three/build
COPY node_modules/@mkkellogg/gaussian-splats-3d/build /srv/node_modules/@mkkellogg/gaussian-splats-3d/build

EXPOSE 6601
