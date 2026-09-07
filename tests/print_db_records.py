import sqlite3
import json

conn = sqlite3.connect('saved_analyses.db')
conn.row_factory = sqlite3.Row

print('=== TABLE SCHEMA ===')
for row in conn.execute('PRAGMA table_info(saved_analyses)'):
    print(f'Col {row["cid"]}: {row["name"]} ({row["type"]}) - NOTNULL: {row["notnull"]}, PK: {row["pk"]}')

total = conn.execute('SELECT COUNT(*) FROM saved_analyses').fetchone()[0]
print(f'\n=== SAVED ANALYSES RECORDS (Total: {total}) ===')
for r in conn.execute('SELECT id, project_name, state, district, project_type, latitude, longitude, delay_probability, risk_tier, predicted_delay_days, composite_risk_score, created_at FROM saved_analyses ORDER BY created_at DESC'):
    print(json.dumps(dict(r), indent=2))
