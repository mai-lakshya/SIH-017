import json

data = json.load(open('dashboard/jk_soi_patch.geojson', encoding='utf-8'))
coords = data['features'][0]['geometry']['coordinates']

rings = []
def get_rings(c):
    if isinstance(c[0][0], (int, float)):
        rings.append(c)
    else:
        for sub in c: get_rings(sub)
get_rings(coords)

lons = [p[0] for r in rings for p in r]
lats = [p[1] for r in rings for p in r]
min_x, max_x = min(lons), max(lons)
min_y, max_y = min(lats), max(lats)

width, height = 600, 600
def to_screen(x, y):
    sx = (x - min_x) / (max_x - min_x) * (width - 40) + 20
    sy = (max_y - y) / (max_y - min_y) * (height - 40) + 20
    return f"{sx:.1f},{sy:.1f}"

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#080D14">']
for r in rings:
    pts = " ".join(to_screen(p[0], p[1]) for p in r)
    svg.append(f'<polygon points="{pts}" fill="#111827" stroke="#06b6d4" stroke-width="2" />')
svg.append('</svg>')

with open('dashboard/jk_rendered_soi.svg', 'w') as f:
    f.write('\n'.join(svg))

print('Wrote dashboard/jk_rendered_soi.svg.')
print(f'Longitude extent: {min_x:.4f} to {max_x:.4f} (Span: {max_x - min_x:.2f} deg)')
print(f'Latitude extent:  {min_y:.4f} to {max_y:.4f} (Span: {max_y - min_y:.2f} deg)')
