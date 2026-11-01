"""
Glacial Lake Monitoring - YOLO + LangChain agent
------------------------------------------------
Upload two satellite photos of the same area (earlier + later).
1. A YOLO segmentation model finds the lake (or glacier) in each photo.
2. The two masks are compared: area that stayed, new area, lost area.
3. A LangChain agent (Groq LLM + one tool) writes the report and answers
   questions about the result. The tool looks up the measured numbers, so the
   LLM never has to guess them.

Needs next to this file:
    bestglacier.pt   (YOLO model; its class is "glacierlake" = lakes)
    glacierlake.pt   (YOLO model; its class is "glacier"     = ice)
Optional: set GROQ_API_KEY to turn on the agent. Without it, the app still
works and writes a plain report.

Run with:  uv run streamlit run glacial_lake_app.py
"""
import hashlib
import html
import os
from datetime import datetime
import area_pages
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import email_alert
import voice_assistant
from dotenv import load_dotenv
load_dotenv()

IMG_SIZE = 640          # both photos are resized to this square size
RISK_LIMIT = 10.0       # |change| above this percent = "Risky"
ALERT_SCORE = 1.0       # risk score ABOVE this sends the email alert automatically
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")   # the agent's LLM

# label shown in the sidebar -> model file
MODELS = {
    "Glacial lake": "bestglacier.pt",
    "Glacier ice": "glacierlake.pt",
}


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_yolo(path):
    """Load a YOLO segmentation model once. Returns None if the file is missing."""
    if not os.path.exists(path):
        return None
    from ultralytics import YOLO
    return YOLO(path)


# ----------------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------------
def decode_image(raw_bytes):
    """Image file bytes (jpg/png) -> RGB image, IMG_SIZE x IMG_SIZE."""
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload(uploaded_file):
    """Uploaded file -> RGB image, IMG_SIZE x IMG_SIZE."""
    return decode_image(uploaded_file.getvalue())


def central_part(mask, radius_frac=0.4):
    """Keep only the detected blobs that touch the middle of the tile, i.e. this lake.
    Other lakes in the same tile are analysed on their own tiles."""
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    middle = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (radius_frac * h / 2) ** 2
    keep = np.unique(labels[middle & (mask == 1)])
    return np.isin(labels, keep[keep > 0]).astype(np.uint8)


def predict_mask(image_rgb, model, conf=0.25):
    """RGB image -> binary mask (1 = detected area, 0 = background)."""
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # Ultralytics expects BGR when given a numpy array (OpenCV convention)
    result = model.predict(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                           conf=conf, imgsz=IMG_SIZE, verbose=False)[0]
    if result.masks is not None:
        for poly in result.masks.xy:          # polygons in image pixel coordinates
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def compare_masks(mask1, mask2):
    """Compare earlier (mask1) and later (mask2) masks."""
    total = float(mask1.size)
    stayed = (mask1 & mask2)                  # in both photos
    gained = (mask2 & (1 - mask1))            # only in later photo
    lost = (mask1 & (1 - mask2))              # only in earlier photo

    area1 = mask1.sum() / total               # fraction of the image (0-1)
    area2 = mask2.sum() / total
    growth_pct = ((area2 - area1) / area1 * 100.0) if area1 > 0 else 0.0

    return {
        "area1": area1,
        "area2": area2,
        "stayed": stayed.sum() / total,
        "gained": gained.sum() / total,
        "lost": lost.sum() / total,
        "growth_pct": growth_pct,
        "status": "Increasing" if area2 > area1 else ("Decreasing" if area2 < area1 else "No change"),
        "risk": "Risky" if abs(growth_pct) > RISK_LIMIT else "Not Risky",
        "risk_score": float(min(abs(growth_pct) / RISK_LIMIT, 10.0)),   # 0-10
        "masks": (stayed, gained, lost),
    }


def make_overlay(image_rgb, masks, alpha=0.5):
    """Paint the later photo: green = stayed, red = gained, blue = lost."""
    stayed, gained, lost = masks
    colour = np.zeros_like(image_rgb)
    colour[stayed == 1] = (0, 255, 0)
    colour[gained == 1] = (255, 0, 0)
    colour[lost == 1] = (0, 0, 255)
    painted = (stayed | gained | lost).astype(bool)
    out = image_rgb.copy()
    out[painted] = cv2.addWeighted(image_rgb, 1 - alpha, colour, alpha, 0)[painted]
    return out


def mask_overlay(image_rgb, mask, alpha=0.5):
    """Show where one mask found something, in green."""
    colour = np.zeros_like(image_rgb)
    colour[mask == 1] = (0, 255, 0)
    out = image_rgb.copy()
    m = mask.astype(bool)
    out[m] = cv2.addWeighted(image_rgb, 1 - alpha, colour, alpha, 0)[m]
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def local_report(r, subject):
    direction = {"Increasing": "grew", "Decreasing": "shrank"}.get(r["status"], "did not change")
    lines = [
        f"**Summary:** the {subject.lower()} area {direction} by {abs(r['growth_pct']):.1f}% between the two images.",
        "",
        f"- Area in first image: {r['area1'] * 100:.2f}% of the frame",
        f"- Area in second image: {r['area2'] * 100:.2f}% of the frame",
        f"- Unchanged area: {r['stayed'] * 100:.2f}%",
        f"- New area: {r['gained'] * 100:.2f}%",
        f"- Lost area: {r['lost'] * 100:.2f}%",
        f"- Risk level: **{r['risk']}** (score {r['risk_score']:.1f}/10)",
    ]
    if r["risk"] == "Risky" and r["status"] == "Increasing" and subject == "Glacial lake":
        lines.append("\nA rapidly growing glacial lake can raise the risk of an outburst flood downstream.")
    elif r["risk"] == "Risky":
        lines.append("\nThis is a large change and worth checking, e.g. for seasonal effects or detection errors.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# LangChain agent (Groq LLM + one tool)
# ----------------------------------------------------------------------------
REPORT_QUESTION = ("Write a short, factual report on this comparison: what changed, "
                   "the areas, the change in percent, and the risk level.")


def result_facts(r):
    """The measurements the agent is allowed to look up (plain text only)."""
    return {
        "area_first": f"{r['area1'] * 100:.2f}% of the image",
        "area_second": f"{r['area2'] * 100:.2f}% of the image",
        "unchanged": f"{r['stayed'] * 100:.2f}% of the image",
        "new_area": f"{r['gained'] * 100:.2f}% of the image",
        "lost_area": f"{r['lost'] * 100:.2f}% of the image",
        "change_percent": f"{r['growth_pct']:+.2f}%",
        "status": r["status"],
        "risk": r["risk"],
        "risk_score": f"{r['risk_score']:.1f} out of 10",
    }

def build_agent(facts, subject):
    """Agent = Groq LLM + the result_details tool (same pattern as text_details)."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def result_details(item: str) -> str:
        """Look up one measurement from the comparison. item must be one of:
        area_first, area_second, unchanged, new_area, lost_area,
        change_percent, status, risk, risk_score."""
        key = item.strip().lower().replace(" ", "_")
        return facts.get(key, f"Unknown item '{item}'. Valid items: {', '.join(facts)}")

    prompt = (
        f"You explain the result of comparing two satellite images of a {subject.lower()}. "
        "Always use the result_details tool to get numbers. Never guess or invent values. "
        "Areas are shares of the image, not square kilometres. "
        "Answer briefly, in plain language."
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)      # reads GROQ_API_KEY itself
    return create_agent(llm, tools=[result_details], system_prompt=prompt)
def ask(agent, question):
    """Send one question to the agent and return its final answer."""
    out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return out["messages"][-1].content


@st.cache_data(show_spinner=False)
def _cached_report(facts, subject):
    # cached, because Streamlit re-runs the page on every click and we don't
    # want a new LLM call each time
    return ask(build_agent(facts, subject), REPORT_QUESTION)


def agent_report(facts, subject):
    """The agent's report, or None if there is no key or anything fails."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        return _cached_report(facts, subject)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Voice assistant (separate agent + tool, see voice_assistant.py)
# ----------------------------------------------------------------------------
def chat_box(placeholder):
    """The chat input with a mic button inside it. Returns (text, audio).
    Needs a recent Streamlit; on an older one it falls back to a chat box
    with a separate voice recorder next to it."""
    try:
        box = st.chat_input(placeholder, accept_audio=True)
    except TypeError:
        chat_col, voice_col = st.columns([3, 2], vertical_alignment="bottom")
        with chat_col:
            text = st.chat_input(placeholder)
        with voice_col:
            audio = st.audio_input("🎙️ Or ask by voice", key="voice_q")
        return text, audio
    if box is None:
        return None, None
    return box.text, box.audio


def voice_answer(audio, facts, subject):
    """Answer a recorded question and show it. Streamlit re-runs the page on
    every click, so the answer is kept in session_state and each recording is
    only sent to the assistant once."""
    if audio is not None:
        data = audio.getvalue()
        audio_id = hashlib.md5(data).hexdigest()
        if st.session_state.get("voice_id") != audio_id:
            with st.spinner("Listening and thinking..."):
                try:
                    st.session_state["voice_result"] = voice_assistant.answer_audio(data, facts, subject)
                except Exception as e:
                    st.session_state["voice_result"] = None
                    st.error(f"The voice assistant could not answer: {e}")
            st.session_state["voice_id"] = audio_id
    result = st.session_state.get("voice_result")
    if audio is not None and result:
        question, answer, speech = result
        if not question:
            st.info("🎙️ I could not hear anything. Please record your question again.")
            return
        with st.container(border=True):
            st.markdown('<span class="voice-tag">🎙️ Voice assistant</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="voice-answer">{html.escape(answer)}</div>', unsafe_allow_html=True)
            if speech:
                st.audio(speech, format="audio/mp3")


# ----------------------------------------------------------------------------
# Email alerts (automatic + manual)
# ----------------------------------------------------------------------------
def send_alert(facts, subject):
    """Send the alert. Uses the LangChain agent if GROQ_API_KEY is set, else a plain message.
    Returns (ok, message)."""
    try:
        if os.environ.get("GROQ_API_KEY"):
            return True, email_alert.send_risk_alert(facts, subject)
        return True, email_alert.send_plain_alert(facts, subject)
    except Exception as e:
        return False, f"Could not send alert: {e}"


def auto_alert(state, key, r, facts, subject, enabled, sender=send_alert):
    """Send the alert ONCE per analysis when risk score > ALERT_SCORE.
    `state` remembers what was already sent, because Streamlit re-runs the
    page on every click and we must not email people again and again.
    Returns (ok, message) or None if no alert was needed/sent."""
    sent = state.setdefault("alerts_sent", {})
    if key in sent:
        return sent[key]
    if enabled and r["risk_score"] > ALERT_SCORE:
        sent[key] = sender(facts, subject)
        return sent[key]
    return None


# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 4rem; max-width: 1100px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #0b3d5c 0%, #1c7ea8 100%);
       color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 1.9rem; padding: 0;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.step {border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
       padding: 16px 18px; height: 100%;}
.step .num {display: inline-block; background: #1c7ea8; color: #fff; width: 26px;
            height: 26px; line-height: 26px; text-align: center; border-radius: 50%;
            font-weight: 700; margin-bottom: 8px;}
.step .t {font-weight: 600; display: block; margin-bottom: 2px;}
.step .d {font-size: .9rem; opacity: .8;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.28);
       padding: 12px 16px; border-radius: 12px;}
.sub-row {padding: 2px 0;}
.voice-tag {display: inline-block; background: rgba(28,126,168,.18); color: #4db8e8;
            font-size: .78rem; font-weight: 600; letter-spacing: .03em;
            padding: 3px 12px; border-radius: 999px; margin-bottom: 10px;}
.voice-answer {font-size: 1.08rem; line-height: 1.65; padding: 2px 0 12px 14px;
               border-left: 3px solid #1c7ea8;}
</style>
"""


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard():
    hero("Glacial Lake Monitoring",
         "Spot fast-growing glacial lakes early, and warn people automatically.")

    n_subs = len(email_alert.list_subscribers())
    c1, c2, c3 = st.columns(3)
    c1.metric("Subscribers", n_subs)
    c2.metric("Alert rule", f"Risk score > {ALERT_SCORE:g}")
    c3.metric("Email mode", "Live" if email_alert.smtp_configured() else "Dry run")

    st.subheader("How it works")
    steps = [
        ("Upload", "Add an earlier and a later satellite image of the same place."),
        ("Detect", "A YOLO model outlines the lake (or glacier) in each image."),
        ("Compare", "The outlines are compared: unchanged, new and lost area."),
        ("Alert", "If the risk score is above the limit, everyone in the list gets an email."),
    ]
    cols = st.columns(4)
    for i, (col, (title, text)) in enumerate(zip(cols, steps), start=1):
        col.markdown(f'<div class="step"><span class="num">{i}</span>'
                     f'<span class="t">{title}</span><span class="d">{text}</span></div>',
                     unsafe_allow_html=True)

    st.write("")
    st.info("Fast-growing glacial lakes can burst their banks and flood the valleys below, "
            "so tracking lake size over time is useful for early warning.")


def page_detection():
    hero("Change Detection", "Compare two images and measure how much the area changed.")

    with st.sidebar:
        st.markdown("**Settings**")
        subject = st.selectbox("What to measure", list(MODELS))
        conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05,
                         help="Lower = detects more (may add false areas). Higher = stricter.")
        auto_on = st.toggle(f"Auto-send email when score > {ALERT_SCORE:g}", value=True)

    model_path = MODELS[subject]
    model = load_yolo(model_path)
    if model is None:
        st.error(f"Could not find `{model_path}`. Put it in the same folder as this app.")
        return

    with st.container(border=True):
        c1, c2 = st.columns(2)
        f1 = c1.file_uploader("Earlier image", type=["jpg", "jpeg", "png"], key="img1")
        f2 = c2.file_uploader("Later image", type=["jpg", "jpeg", "png"], key="img2")
    if not (f1 and f2):
        st.info("Upload both images to start.")
        return

    img1, img2 = read_upload(f1), read_upload(f2)
    with st.spinner("Segmenting..."):
        mask1 = predict_mask(img1, model, conf)
        mask2 = predict_mask(img2, model, conf)
    r = compare_masks(mask1, mask2)
    facts = result_facts(r)

    if mask1.sum() == 0:
        st.warning("Nothing was detected in the first image, so a percentage change can't be computed. "
                   "Try lowering the confidence slider.")

    # identifies THIS analysis, so the automatic email is only sent once for it
    key = hashlib.md5(f1.getvalue() + f2.getvalue()).hexdigest() + f"|{subject}|{conf}"
    fresh = key not in st.session_state.get("alerts_sent", {})
    with st.spinner("Sending alert emails..."):
        alert = auto_alert(st.session_state, key, r, facts, subject, auto_on)
    if alert and fresh:
        st.toast("Alert emails sent" if alert[0] else "Alert failed", icon="📧" if alert[0] else "⚠️")

    # ---- summary cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Area change", f"{r['growth_pct']:+.2f}%")
    m2.metric("Status", r["status"])
    m3.metric("Risk score", f"{r['risk_score']:.1f} / 10")
    m4.metric("Risk level", r["risk"])

    tab_maps, tab_report, tab_alert = st.tabs(["Maps", "Report & questions", "Email alert"])

    with tab_maps:
        a, b = st.columns(2)
        a.image(mask_overlay(img1, mask1), caption=f"{subject} - first image")
        b.image(mask_overlay(img2, mask2), caption=f"{subject} - second image")
        st.image(make_overlay(img2, r["masks"]),
                 caption="Green = unchanged, Red = new area, Blue = lost area")
        st.caption(f"Unchanged {r['stayed'] * 100:.2f}%  |  New {r['gained'] * 100:.2f}%  |  "
                   f"Lost {r['lost'] * 100:.2f}%  (share of the image)")

    with tab_report:
        st.markdown(agent_report(facts, subject) or local_report(r, subject))
        if os.environ.get("GROQ_API_KEY"):
            st.divider()
            question, audio = chat_box("Ask about this result, e.g. Is it dangerous?")
            if question:
                with st.spinner("Thinking..."):
                    try:
                        st.markdown(ask(build_agent(facts, subject), question))
                    except Exception as e:
                        st.error(f"The assistant could not answer: {e}")
            voice_answer(audio, facts, subject)
        else:
            st.caption("Set GROQ_API_KEY to get an AI-written report and ask questions.")

    with tab_alert:
        st.write(f"Rule: an email goes to all subscribers when the risk score is above "
                 f"**{ALERT_SCORE:g}**. This result scored **{r['risk_score']:.1f}**.")
        if alert:
            (st.success if alert[0] else st.error)(alert[1])
        elif r["risk_score"] > ALERT_SCORE:
            st.warning("Risk is above the limit, but automatic alerts are switched off in the sidebar.")
        else:
            st.success("Risk is below the limit, so no alert was needed.")
        if st.button("Send alert now (manual)"):
            with st.spinner("Sending..."):
                ok, msg = send_alert(facts, subject)
            (st.success if ok else st.error)(msg)


def page_subscribers():
    hero("Subscribers", "People who receive the email alerts.")

    with st.container(border=True):
        st.markdown("**Add a subscriber**")
        with st.form("add_subscriber", clear_on_submit=True):
            c1, c2, c3 = st.columns([3, 4, 1.4], vertical_alignment="bottom")
            name = c1.text_input("Name")
            email = c2.text_input("Email")
            submitted = c3.form_submit_button("Add", use_container_width=True)
        if submitted:
            ok, msg = email_alert.add_subscriber(name, email)
            (st.success if ok else st.error)(msg)

    people = email_alert.list_subscribers()
    st.subheader(f"All subscribers ({len(people)})")
    if not people:
        st.info("No subscribers yet. Add one above.")
    for sub_id, sub_name, sub_email in people:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 4, 1.4], vertical_alignment="center")
            c1.write(f"**{sub_name}**")
            c2.write(sub_email)
            if c3.button("Remove", key=f"del_{sub_id}", use_container_width=True):
                email_alert.delete_subscriber(sub_id)
                st.rerun()


# ----------------------------------------------------------------------------
# Region monitoring: region name -> every lake -> YOLO before/after -> score -> report -> alerts
# (satellite/geocoding/scoring/email helpers are in lake_pipeline.py, YOLO is used here)
# ----------------------------------------------------------------------------
DETECT_DAYS = 90        # how far back to look for the recent image that locates the lakes


def analyse_lake(lp, model, lake, before_year, after_year, conf):
    """ONE lake: fetch its before/after images, run YOLO on each, measure the change, score it."""
    row = {k: lake[k] for k in ("id", "name", "key", "lon", "lat", "history_ha")}
    try:
        raw1, raw2, half_km = lp.fetch_lake_images(lake["lon"], lake["lat"], lake["area_ndwi_ha"],
                                                   before_year, after_year)
        img1, img2 = decode_image(raw1), decode_image(raw2)
        mask1 = central_part(predict_mask(img1, model, conf))
        mask2 = central_part(predict_mask(img2, model, conf))
        r = compare_masks(mask1, mask2)
    except Exception as e:
        return {**row, "error": str(e)[:200], "risk_score": None}
    tile_ha = (2 * half_km) ** 2 * 100                      # ground area of the picture in hectares
    before_ha, after_ha = float(r["area1"] * tile_ha), float(r["area2"] * tile_ha)
    total, flags = lp.yolo_score(before_ha, after_ha, lake["series"], lake["area_ndwi_ha"])
    return {**row,
            "before_ha": round(before_ha, 1), "after_ha": round(after_ha, 1),
            "change_pct": round((after_ha - before_ha) / before_ha * 100, 1) if before_ha > 0 else None,
            "risk_score": round(total), "risk_label": lp.risk_label(total), "notes": "; ".join(flags),
            "imgs": (img1, img2), "masks": (mask1, mask2),
            "change_overlay": make_overlay(img2, r["masks"])}


def analyse_region(lp, model, region, radius, before_year, after_year, conf, max_lakes):
    """The whole workflow. The report and the emails are only built after EVERY lake is done."""
    bar = st.progress(0.0, text="Looking up the region...")
    lon, lat, place = lp.geocode(region)
    bar.progress(0.05, text="Finding the glacial lakes...")
    lakes, date = lp.find_lakes(lon, lat, radius, DETECT_DAYS, max_lakes)
    results, warnings = [], []
    for i, lake in enumerate(lakes):
        bar.progress(0.1 + 0.75 * i / len(lakes), text=f"Analysing {lake['name']} ({i + 1}/{len(lakes)})...")
        results.append(analyse_lake(lp, model, lake, before_year, after_year, conf))
    scored = [r for r in results if r.get("risk_score") is not None]
    bar.progress(0.9, text="Looking for places downstream...")
    try:
        places = lp.find_places(scored)
    except Exception as e:
        places = {}
        warnings.append(f"Could not find downstream places: {str(e)[:150]}")
    bar.progress(0.97, text="Writing the report...")
    years = (before_year, after_year)
    report, alerts = lp.region_alerts(region, results, places, years)
    for a in alerts:
        a["status"] = "planned (dry run)"
    bar.empty()
    return {"region": region, "place": place, "lon": lon, "lat": lat, "image_date": date,
            "years": years, "lakes": results, "places": places, "report": report,
            "alerts": alerts, "summary": {}, "warnings": warnings, "dry_run": True}


def page_region():
    import lake_pipeline as lp     # imported here so the rest of the app still starts if it fails

    hero("Region Monitoring",
         "Name a region: every glacial lake in it is compared before and after with YOLO, scored, and reported.")

    with st.container(border=True):
        c1, c2 = st.columns([4, 2])
        region = c1.text_input("Region name", placeholder="e.g. South Lhonak Lake, Sikkim")
        radius = c2.slider("Search radius (km)", 1.0, float(lp.MAX_RADIUS_KM), 10.0, 0.5)
        c3, c4, c5, c6 = st.columns(4)
        last_year = datetime.now().year - 1
        before_year = c3.number_input("Before year", 1985, last_year, 1990)
        after_year = c4.number_input("After year", 1985, last_year, 2020)
        conf = c5.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
        max_lakes = c6.slider("Max lakes", 1, 20, 8, help="The largest lakes are analysed first.")
        run = st.button("Analyse region", type="primary")
        st.caption("Takes a few minutes. Emails are never sent without your confirmation. "
                   "The search covers a circle around the named place, so for a big region "
                   "use a specific valley or lake name.")

    if run:
        model = load_yolo(MODELS["Glacial lake"])
        if not region.strip():
            st.error("Please enter a region name.")
        elif before_year >= after_year:
            st.error("The before year must be earlier than the after year.")
        elif model is None:
            st.error(f"Could not find `{MODELS['Glacial lake']}`. Put it in the same folder as this app.")
        else:
            try:
                st.session_state["region_result"] = analyse_region(
                    lp, model, region.strip(), radius, int(before_year), int(after_year), conf, max_lakes)
            except Exception as e:
                st.session_state.pop("region_result", None)
                st.error(f"Analysis failed: {e}")

    res = st.session_state.get("region_result")
    if not res:
        st.info("Enter a region name and click Analyse region.")
        return

    st.caption(f"'{res['region']}' = {res['place']} ({res['lat']:.3f}, {res['lon']:.3f}). "
               f"Newest satellite image used to locate lakes: {res['image_date']}.")
    for w in res["warnings"]:
        st.warning(w)
    lakes = res["lakes"]
    if not lakes:
        st.info("No glacial lakes were found around this place. Try another name or a larger radius.")
        return

    scored = [l for l in lakes if l.get("risk_score") is not None]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lakes analysed", len(scored))
    for col, level in zip((m2, m3, m4), ("High", "Medium", "Low")):
        col.metric(f"{level} risk", sum(1 for l in scored if l["risk_label"] == level))
    st.map(pd.DataFrame(lakes)[["lat", "lon"]])

    st.subheader("All lakes")
    if scored:
        table = pd.DataFrame(scored)[["name", "risk_label", "risk_score", "before_ha",
                                      "after_ha", "change_pct", "notes"]]
        st.dataframe(table.sort_values("risk_score", ascending=False), hide_index=True)

    st.subheader("Lake by lake")
    before_y, after_y = res["years"]
    for lk in sorted(lakes, key=lambda l: -1 if l.get("risk_score") is None else l["risk_score"], reverse=True):
        if lk.get("risk_score") is None:
            with st.expander(f"{lk['name']}: not analysed"):
                st.error(lk["error"])
            continue
        with st.expander(f"{lk['name']}: {lk['risk_label']} (score {lk['risk_score']}/100)",
                         expanded=lk["risk_label"] == "High"):
            a, b, c = st.columns(3)
            a.image(mask_overlay(lk["imgs"][0], lk["masks"][0]), caption=f"Before ({before_y}): YOLO detection")
            b.image(mask_overlay(lk["imgs"][1], lk["masks"][1]), caption=f"After ({after_y}): YOLO detection")
            c.image(lk["change_overlay"], caption="Green = unchanged, Red = new area, Blue = lost area")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(f"Area {before_y}", f"{lk['before_ha']} ha")
            k2.metric(f"Area {after_y}", f"{lk['after_ha']} ha")
            k3.metric("Change", f"{lk['change_pct']:+.0f}%" if lk["change_pct"] is not None else "new lake")
            k4.metric("Risk", f"{lk['risk_label']} ({lk['risk_score']})")
            if lk["history_ha"]:
                st.caption(f"Inventory history (ha): {lk['history_ha']}")
            if lk["notes"]:
                st.caption(f"Notes: {lk['notes']}")
            places = res["places"].get(lk["key"], [])
            if places:
                st.caption("Downstream: " + ", ".join(f"{p['place']} ({p['river_km']} km)" for p in places[:8]))

    st.subheader("Summary report and alerts")
    with st.expander("Summary report (this is what authorities receive)", expanded=True):
        st.text(res["report"])
    alerts = res["alerts"]
    if not alerts:
        st.info("No emails planned. Authorities are emailed when a lake is Medium or High, and residents "
                "when a High lake is above their place. Register people on the Alert Contacts page.")
        return
    st.dataframe(pd.DataFrame(alerts)[["level", "to_name", "to_email", "role", "status"]], hide_index=True)
    if res["dry_run"]:
        st.warning("Dry run: nothing has been sent.")
        if not (os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")):
            st.info("Email is not set up (SMTP_USER / SMTP_PASSWORD missing in .env), so sending is disabled.")
            return
        with st.form("region_send_form"):
            confirm = st.text_input("To really send these emails, type SEND here")
            go = st.form_submit_button("Send alerts now")
        if go:
            if confirm.strip() == "SEND":
                res["summary"] = lp.deliver(alerts)
                res["dry_run"] = False
                st.rerun()
            else:
                st.error("Type SEND in the box first.")
    else:
        st.success(f"Done: {res['summary']}")


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------
PAGES = {
    "🏠  Dashboard": page_dashboard,
    "🛰️  Change Detection": page_detection,
    "🧭  Region Monitoring": page_region,
    "👥  Subscribers": page_subscribers,
    "📇  Alert Contacts": area_pages.page_area_contacts,
}


def main():
    st.set_page_config(page_title="Glacial Lake Monitoring", page_icon="🏔️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🏔️ GlacierWatch")
        page = st.radio("Menu", list(PAGES), label_visibility="collapsed")
        st.divider()
    PAGES[page]()


if __name__ == "__main__":
    main()
