# GlacierWatch (cleaned project)

Glacial-lake outburst-risk screening: compare two satellite images with YOLO,
or monitor a whole region (Google Earth Engine + YOLO), score each lake, find villages
downstream, and email alerts to registered contacts.

Run: `uv run streamlit run glacial_lake_app.py`

## Region Monitoring (new page)

1. Enter a region name (best: a valley or lake name; the search is a circle of up to 20 km around it).
2. Lakes are located (recent Sentinel-2 water detection, matched to the 2020 inventory, named from OpenStreetMap when possible; the largest N are kept).
3. For each lake, one at a time: a Landsat true-colour image for the before year and the after year is fetched, YOLO (`bestglacier.pt`) outlines the lake in both, the area change is measured, and a 0-100 score is computed.
   Score = YOLO change (40) + lake size (20) + long-term inventory growth (20) + lake type (20). Low < 30, Medium 30-59, High >= 60.
4. Only after every lake is finished: the summary report is written and emails are planned (dry run first).
   Authorities (place blank, the region name, or a downstream place) get the full report when any lake is Medium/High. Residents of a downstream place get a short notice when a High lake is above them. Sending needs SMTP in `.env` and typing SEND.

## Files and their purpose

| File | Purpose |
|---|---|
| `glacial_lake_app.py` | **Entry point.** Streamlit app: Dashboard, Change Detection (YOLO on two uploaded images, report, chat, voice, auto email), **Region Monitoring (YOLO on satellite before/after images of every lake in a region)**, Subscribers, plus the Alert Contacts page from `area_pages.py`. All YOLO code lives here. |
| `area_pages.py` | The "Alert Contacts" page. |
| `lake_pipeline.py` | Satellite/data engine: Sentinel-2 lake detection (Earth Engine), 1990-2020 inventory history, downstream villages (OSM + HydroSHEDS), and email sending. Section 5 supports Region Monitoring: geocoding, lake names, Landsat before/after images, the YOLO-based risk score and the region report/recipients. |
| `contacts_db.py` | SQLite storage (`contacts.db`) for authority/resident contacts and the alert log (prevents repeat emails within 24 h). |
| `email_alert.py` | Subscribers (`subscribers.db`), SMTP sending (dry run unless SMTP is configured), and the LLM/plain alert used by Change Detection. |
| `voice_assistant.py` | Voice questions: Groq Whisper speech-to-text, LangChain agent, optional gTTS spoken answer. |
| `uv.lock` | Locked dependency versions (keep with your `pyproject.toml`). |

## Needed at runtime (not code, keep your own copies)

- `pyproject.toml`, `.env` (GROQ_API_KEY, SMTP_USER, SMTP_PASSWORD; never commit it)
- `bestglacier.pt`, `glacierlake.pt` (YOLO models)
- `GlacialLake_20220726/` (inventory shapefiles)
- Auto-created: `contacts.db`, `subscribers.db`, `osm_places_*.json`

## Removed

- Superseded by `lake_pipeline.py`: `detect_lakes.py`, `risk_score.py`, `lake_history.py`, `affected_places.py`, `send_alerts.py`, `fetch_lake.py`, `analyze_lake.py`
- Dev / validation only: `ee_test.py`, `inspect_shapefiles.py`, `backtest.py`, `threshold_test.py`
- Old duplicate UI: `glacier_dashboard.py`
- Dummy-data scripts: `create_db.py`, `update_email.py`
- Optional, unused by the app (restorable): `contacts_api.py` (REST API), `scheduled_run.py` (unattended runs). Their dependencies (`fastapi`, `uvicorn`) were removed from `pyproject.toml`.
- Area Risk Screening feature: its page, `run_area`/`score_lakes`/baseline/alert-planning code in `lake_pipeline.py`, the command-line entry point, and `lake_tool.py` (the chat assistant's area-assessment tool).
- Dead code: `contacts_for_alert()` in `contacts_db.py`, the demo `__main__` block in `email_alert.py`.
