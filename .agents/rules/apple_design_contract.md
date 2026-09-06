# Project Design Contract Directives (DESIGN.md)

This project has an authoritative design system specified in [DESIGN.md](../../DESIGN.md) based on the Apple Style Reference.

## Binding Principles
1. **Palette Restraint**: Canvas is `#f5f5f7`, text is `#1d1d1f`, filled action is `#0071e3`, outlined action is `#0066cc`, decorative line is `#2997ff`.
2. **Typography**: SF Pro Display / SF Pro Text (or system-ui / Inter fallback). Strict negative tracking (-0.016em body, tighter as size increases).
3. **Pills**: All action buttons and tags MUST use `980px` border-radius (`--radius-buttons: 980px`). Cards and containers use `8px`.
4. **No Shadows on Cards**: Avoid box-shadows on cards, containers, and buttons. Elevation is expressed via subtle surface shifts (`#f5f5f7` canvas, `#f4f8fb` wash, `#e2e2e5` pebble) and 1px hairline borders (`#d2d2d7`).
5. **Fluid Motion**: Align motion with `apple-design` skill (instant pointer-down response, velocity inheritance, interruptible spring curves).
