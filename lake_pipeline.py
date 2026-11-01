"""Glacial lake pipeline: lake detection, inventory history, downstream places, email alerts, region monitoring."""
import json, math, os, smtplib, ssl, time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import ee, requests, shapefile
from dotenv import load_dotenv
from pyproj import CRS, Transformer
from shapely.geometry import Point, box, shape

import contacts_db as db

load_dotenv()

PROJECT_ID = "gen-lang-client-0537580453"
INVENTORY_FOLDER = "GlacialLake_20220726"
INVENTORY_YEARS = [1990, 2000, 2010, 2015, 2020]

# ---- lake detection ----
MAX_RADIUS_KM = 20
MAX_CLOUD = 30
NDWI_MIN = 0.05
NIR_MAX = 1500
SLOPE_MAX = 15
MIN_AREA_HA = 1.0
CLOSE_PX = 2
HOLE_MAX_PX = 100

# ---- scoring / matching ----
START_KM = 1.2
STEP_KM = 0.4

# ---- downstream places ----
PLACES_MIN_SCORE = 30
DOWNSTREAM_KM = 120
PLACE_RADIUS_KM = 110
HAND_MAX_M = 60
MIN_DROP_M = 100
PLACE_TYPES = "village|hamlet|town|city"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "glacier-lake-risk-student-project/0.1", "Accept": "*/*"}

# ---- alerts ----
REPEAT_HOURS = 24
MAX_PLACES = 8

KM_PER_DEG = 111.19
_ee_ready = False


def _init_ee():
    global _ee_ready
    if not _ee_ready:
        ee.Initialize(project=PROJECT_ID)
        _ee_ready = True


def dist_km(lon1, lat1, lon2, lat2):
    dx = (lon1 - lon2) * KM_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    dy = (lat1 - lat2) * KM_PER_DEG
    return math.hypot(dx, dy)


def clamp01(x):
    return max(0.0, min(1.0, x))


def nearest(lakes, lon, lat, max_km):
    best, best_d = None, max_km
    for lk in lakes:
        d = dist_km(lon, lat, lk["lon"], lk["lat"])
        if d <= best_d:
            best, best_d = lk, d
    return best


# =====================================================================
# 1. Lake detection (satellite, via Earth Engine)
# =====================================================================

def _detect(lon, lat, radius_km, start, end, max_cloud):
    """Lakes from Sentinel-2 images between two dates (YYYY-MM-DD). Returns (lakes, newest image date)."""
    if radius_km > MAX_RADIUS_KM:
        raise ValueError(f"radius_km must be at most {MAX_RADIUS_KM}")
    _init_ee()
    region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000).bounds()
    inner = region.buffer(-20)

    def mask_clouds(img):
        scl = img.select("SCL")
        bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        return img.updateMask(bad.Not())

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(mask_clouds))
    if col.size().getInfo() == 0:
        raise RuntimeError(f"No usable satellite images between {start} and {end}. "
                           "Try a larger 'days' value.")
    date = ee.Date(col.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()

    comp = col.median().clip(region)
    green, nir = comp.select("B3"), comp.select("B8")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).clip(region)
    raw = ndwi.gt(NDWI_MIN).And(nir.lt(NIR_MAX)).And(slope.lt(SLOPE_MAX)).unmask(0)
    closed = (raw.focalMax(radius=CLOSE_PX, kernelType="circle", units="pixels")
                 .focalMin(radius=CLOSE_PX, kernelType="circle", units="pixels"))
    hole_size = closed.connectedPixelCount(maxSize=HOLE_MAX_PX, eightConnected=True)
    holes = closed.eq(0).And(hole_size.lt(HOLE_MAX_PX))
    water = closed.Or(holes).selfMask().rename("water")

    vectors = water.reduceToVectors(
        geometry=region, scale=10, geometryType="polygon",
        eightConnected=True, maxPixels=1e9, bestEffort=True)

    def add_props(f):
        g = f.geometry()
        c = g.centroid(1).coordinates()
        return f.set({"area_ha": g.area(1).divide(10000),
                      "lon": c.get(0), "lat": c.get(1),
                      "edge": ee.Algorithms.If(inner.contains(g, 1), 0, 1)})

    vectors = vectors.map(add_props).filter(ee.Filter.gte("area_ha", MIN_AREA_HA))
    fc = vectors.getInfo()
    lakes = []
    for f in fc["features"]:
        p = f["properties"]
        lakes.append({"area_ha": round(p["area_ha"], 2), "lon": round(p["lon"], 5),
                      "lat": round(p["lat"], 5), "edge": int(p["edge"])})
    lakes.sort(key=lambda x: x["area_ha"], reverse=True)
    for i, lk in enumerate(lakes, 1):
        lk["id"] = i
    return lakes, date


def detect_lakes(lon, lat, radius_km=5.0, days=30):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return _detect(lon, lat, radius_km, start.strftime("%Y-%m-%d"),
                   end.strftime("%Y-%m-%d"), MAX_CLOUD)

# =====================================================================
# 2. Inventory history (1990-2020 shapefiles)
# =====================================================================
def inventory_history(lon, lat, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    corners = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
               (lon + dlon, lat + dlat), (lon - dlon, lat + dlat)]
    hist = {}
    for year in INVENTORY_YEARS:
        shp = os.path.join(INVENTORY_FOLDER, f"GlacialLake_{year}.shp")
        prj = shp[:-4] + ".prj"
        if not (os.path.exists(shp) and os.path.exists(prj)):
            continue
        with open(prj) as fh:
            crs = CRS.from_wkt(fh.read())
        to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = zip(*[to_proj.transform(x, y) for x, y in corners])
        aoi = box(min(xs), min(ys), max(xs), max(ys))
        found = []
        for sr in shapefile.Reader(shp).iterShapeRecords():
            if sr.shape.shapeType == 0:
                continue
            b = sr.shape.bbox
            if b[2] < aoi.bounds[0] or b[0] > aoi.bounds[2] or b[3] < aoi.bounds[1] or b[1] > aoi.bounds[3]:
                continue
            g = shape(sr.shape.__geo_interface__)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.intersects(aoi):
                continue
            rec = sr.record.as_dict()
            c = g.centroid
            lo, la = to_lonlat.transform(c.x, c.y)
            found.append({"lon": lo, "lat": la,
                          "area_ha": float(rec.get("Area") or 0) * 100,   # Area field is km2
                          "type": rec.get("Type") or ""})
        hist[year] = found
    return hist


def history_for(hist, lon, lat):
    series = {}
    cur = nearest(hist.get(2020, []), lon, lat, START_KM)
    if cur is None:
        return series
    series[2020] = cur
    for y in (2015, 2010, 2000, 1990):
        m = nearest(hist.get(y, []), cur["lon"], cur["lat"], STEP_KM)
        if m is not None:
            series[y] = m
            cur = m
    return series


def type_points(t):
    t = t.lower()
    if t.startswith("pro"):
        return 20
    if t.startswith("supra"):
        return 10
    return 5

def _group_by_inventory(detected, hist):
    groups = {}
    for d in detected:
        series = history_for(hist, d["lon"], d["lat"])
        key = (series[2020]["lon"], series[2020]["lat"]) if 2020 in series else ("new", d["id"])
        groups.setdefault(key, {"series": series, "parts": []})["parts"].append(d)
    return groups


# =====================================================================
# 3. Places along the river below each lake
# =====================================================================
def fetch_places(lon, lat, radius_km):
    cache = f"osm_places_{lat:.2f}_{lon:.2f}_{radius_km}km.json"
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    q = (f'[out:json][timeout:180];'
         f'node["place"~"^({PLACE_TYPES})$"](around:{radius_km * 1000},{lat},{lon});'
         f'out body;')
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                resp = requests.post(url, data={"data": q}, headers=HEADERS, timeout=200)
                if resp.status_code in (429, 502, 504):
                    time.sleep(30 * (attempt + 1))
                    continue
                resp.raise_for_status()
                elements = resp.json()["elements"]
                with open(cache, "w", encoding="utf-8") as fh:
                    json.dump(elements, fh)
                return elements
            except Exception:
                time.sleep(5)
    raise RuntimeError("Could not get place data from OpenStreetMap (try again later).")


def _terrain():
    return (ee.Image("USGS/SRTMGL1_003").rename("elev")
            .addBands(ee.Image("MERIT/Hydro/v1_0_1").select("hnd")))


def sample_terrain(points):
    out, terrain = [], _terrain()
    for i in range(0, len(points), 1000):
        chunk = points[i:i + 1000]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lo, la]), {"k": j}) for j, (lo, la) in enumerate(chunk)])
        res = terrain.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30).getInfo()
        vals = {f["properties"]["k"]: f["properties"] for f in res["features"]}
        out.extend(vals.get(j, {}) for j in range(len(chunk)))
    return out


def lake_elevation(lon, lat):
    pt = ee.Geometry.Point([lon, lat]).buffer(100)
    return ee.Image("USGS/SRTMGL1_003").reduceRegion(ee.Reducer.mean(), pt, 30).get("elevation").getInfo()


def downstream_chain(lon, lat, cache):
    basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_10")
    start = ee.Feature(basins.filterBounds(ee.Geometry.Point([lon, lat])).first()).getInfo()
    sid = start["properties"]["HYBAS_ID"]
    if sid in cache:
        return cache[sid]
    dist0 = start["properties"]["DIST_SINK"]
    chain = [{"geom": shape(start["geometry"]).buffer(0), "river_km": 0.0}]
    cur = start
    while True:
        nxt = cur["properties"].get("NEXT_DOWN")
        if not nxt:
            break
        f = ee.Feature(basins.filter(ee.Filter.eq("HYBAS_ID", nxt)).first()).getInfo()
        travelled = dist0 - f["properties"]["DIST_SINK"]
        if travelled > DOWNSTREAM_KM:
            break
        chain.append({"geom": shape(f["geometry"]).buffer(0), "river_km": round(travelled, 1)})
        cur = f
    cache[sid] = chain
    return chain


def find_places(rows):
    """Returns {lake key: [places below the lake, nearest along the river first]}."""
    targets = [r for r in rows if r["risk_score"] >= PLACES_MIN_SCORE]
    if not targets:
        return {}
    _init_ee()
    clon = sum(r["lon"] for r in targets) / len(targets)
    clat = sum(r["lat"] for r in targets) / len(targets)
    elements = fetch_places(clon, clat, PLACE_RADIUS_KM + 15)
    named = [p for p in elements if p.get("tags", {}).get("name") or p.get("tags", {}).get("name:en")]
    for p, s in zip(named, sample_terrain([(p["lon"], p["lat"]) for p in named])):
        p["elev"], p["hnd"] = s.get("elev"), s.get("hnd")

    result, cache = {}, {}
    for lk in targets:
        try:
            lake_elev = lake_elevation(lk["lon"], lk["lat"])
            chain = downstream_chain(lk["lon"], lk["lat"], cache)
        except Exception as e:
            print(f"  (could not trace the river for lake {lk['id']}: {str(e)[:80]})")
            continue
        out = []
        for p in named:
            if p.get("elev") is None or p.get("hnd") is None:
                continue
            if lake_elev - p["elev"] < MIN_DROP_M or p["hnd"] > HAND_MAX_M:
                continue
            pt = Point(p["lon"], p["lat"])
            hit = next((c for c in chain if c["geom"].contains(pt)), None)
            if hit is None:
                continue
            tags = p["tags"]
            out.append({"place": tags.get("name:en") or tags.get("name"), "type": tags.get("place"),
                        "river_km": hit["river_km"],
                        "straight_km": round(dist_km(lk["lon"], lk["lat"], p["lon"], p["lat"]), 1),
                        "elevation_m": round(p["elev"]), "height_above_river_m": round(p["hnd"]),
                        "lon": round(p["lon"], 5), "lat": round(p["lat"], 5),
                        "population": tags.get("population", "")})
        out.sort(key=lambda r: (r["river_km"], r["straight_km"]))
        result[lk["key"]] = out
    return result


# =====================================================================
# 4. Alerts
# =====================================================================
def resident_message(lake, place):
    subject = f"Safety notice: glacial lake above {place['place']}"
    body = (
        f"Hello,\n\nAn automated satellite check has marked a glacial lake upstream of "
        f"{place['place']} (about {place['river_km']} km along the river) as higher risk, "
        "mainly because it has grown over the past decades.\n\n"
        "This is NOT a flood warning and it does not mean a flood is expected. "
        "Please follow instructions from your local disaster management authority and "
        "keep their emergency contacts handy.\n\n"
        "To stop receiving these messages, contact the sender of this email.\n")
    return subject, body


def send_email(to_addr, subject, body):
    user, password = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    if not user or not password:
        raise RuntimeError("SMTP_USER / SMTP_PASSWORD missing in the .env file")
    msg = EmailMessage()
    msg["From"] = os.getenv("FROM_EMAIL", user)
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"),
                      int(os.getenv("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, password)
        s.send_message(msg)


def deliver(alerts):
    sent = skipped = failed = 0
    for a in alerts:
        if db.already_sent(a["lake_key"], a["to_email"], a["level"], REPEAT_HOURS):
            a["status"] = "skipped (already alerted recently)"
            skipped += 1
            continue
        try:
            send_email(a["to_email"], a["subject"], a["body"])
            db.log_alert(a["lake_key"], a["to_email"], a["level"])
            a["status"] = "sent"
            sent += 1
        except Exception as e:
            a["status"] = f"FAILED: {str(e)[:80]}"
            failed += 1
    return {"sent": sent, "skipped": skipped, "failed": failed}


# =====================================================================
# 5. Region monitoring: region name -> lakes -> before/after images -> YOLO score -> report
#    (the YOLO step itself lives in glacial_lake_app.py; this part has no YOLO code)
# =====================================================================
LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2}


def risk_label(total):
    return "High" if total >= 60 else "Medium" if total >= 30 else "Low"


def geocode(place):
    """Region name -> (lon, lat, full name) using OpenStreetMap Nominatim."""
    resp = requests.get("https://nominatim.openstreetmap.org/search",
                        params={"q": place, "format": "json", "limit": 1},
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        raise ValueError(f"Could not find a place called '{place}'. Try a more specific name.")
    h = hits[0]
    return float(h["lon"]), float(h["lat"]), h["display_name"]


def lake_names(lon, lat, radius_km):
    """Named water bodies from OpenStreetMap: [(lon, lat, name)]. Empty list if unavailable."""
    q = (f'[out:json][timeout:60];nwr["natural"="water"]["name"]'
         f'(around:{radius_km * 1000},{lat},{lon});out center tags;')
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": q}, headers=HEADERS, timeout=90)
            resp.raise_for_status()
            out = []
            for e in resp.json()["elements"]:
                c = e.get("center") or e
                if "lat" in c:
                    out.append((c["lon"], c["lat"], e["tags"].get("name:en") or e["tags"]["name"]))
            return out
        except Exception:
            continue
    return []


def find_lakes(lon, lat, radius_km, days=30, max_lakes=8):
    """The glacial lakes around a point: satellite water detection, grouped by the 2020
    inventory lake each piece belongs to. Returns (lakes largest first, newest image date)."""
    detected, date = detect_lakes(lon, lat, radius_km, days)
    hist = inventory_history(lon, lat, radius_km)
    lakes = []
    for g in _group_by_inventory(detected, hist).values():
        parts, series = g["parts"], g["series"]
        area = sum(p["area_ha"] for p in parts)
        lakes.append({
            "lon": round(sum(p["lon"] * p["area_ha"] for p in parts) / area, 5),
            "lat": round(sum(p["lat"] * p["area_ha"] for p in parts) / area, 5),
            "area_ndwi_ha": round(area, 1), "series": series,
            "history_ha": " | ".join(f"{y}: {series[y]['area_ha']:.1f}" for y in sorted(series))})
    lakes.sort(key=lambda l: l["area_ndwi_ha"], reverse=True)
    lakes = lakes[:max_lakes]
    names = lake_names(lon, lat, radius_km) if lakes else []
    for i, lk in enumerate(lakes, 1):
        lk["id"] = i
        lk["key"] = f"{lk['lat']:.2f},{lk['lon']:.2f}"
        near = min(names, key=lambda n: dist_km(lk["lon"], lk["lat"], n[0], n[1]), default=None)
        close = near and dist_km(lk["lon"], lk["lat"], near[0], near[1]) <= 1.5
        lk["name"] = near[2] if close else f"Lake {i}"
    for lk in lakes:                                   # two lakes with one OSM name
        if sum(1 for o in lakes if o["name"] == lk["name"]) > 1:
            lk["name"] += f" #{lk['id']}"
    return lakes, date


def _landsat_year(region, year):
    """Cloud-free Landsat true-colour composite (bands red/green/blue, 0-1) for one year."""
    if year <= 2011:                  # Landsat 5
        ids, bands = ["LANDSAT/LT05/C02/T1_L2"], ["SR_B3", "SR_B2", "SR_B1"]
    elif year == 2012:                # Landsat 7 (the gap year between 5 and 8)
        ids, bands = ["LANDSAT/LE07/C02/T1_L2"], ["SR_B3", "SR_B2", "SR_B1"]
    else:                             # Landsat 8 / 9
        ids, bands = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"], ["SR_B4", "SR_B3", "SR_B2"]
    base = ee.ImageCollection(ids[0])
    for extra in ids[1:]:
        base = base.merge(ee.ImageCollection(extra))
    base = base.filterBounds(region)

    def prep(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(0b11110).eq(0)      # no cloud, cirrus or shadow
        rgb = img.select(bands, ["red", "green", "blue"]).multiply(0.0000275).add(-0.2)
        return rgb.updateMask(clear)

    # post-monsoon autumn has the least cloud and snow; widen to the neighbour years if empty
    for start, end in ((f"{year}-09-01", f"{year}-12-15"), (f"{year - 1}-09-01", f"{year + 1}-12-15")):
        col = base.filterDate(start, end)
        if col.size().getInfo() > 0:
            return col.map(prep).median().clip(region)
    raise RuntimeError(f"No usable Landsat images for {year} here.")


def fetch_lake_images(lon, lat, area_ha, before_year, after_year):
    """Two true-colour PNGs (bytes) of one lake, before and after. Returns (before, after, half_km)."""
    _init_ee()
    half_km = max(2.0, 2.5 * math.sqrt(area_ha / 100.0))      # the box grows with the lake
    region = ee.Geometry.Point([lon, lat]).buffer(half_km * 1000).bounds()
    out = []
    for year in (before_year, after_year):
        url = _landsat_year(region, year).getThumbURL({
            "region": region, "dimensions": 640, "format": "png",
            "min": 0.0, "max": 0.3, "gamma": 1.4})
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out.append(resp.content)
    return out[0], out[1], half_km


def yolo_score(before_ha, after_ha, series, fallback_ha):
    """Risk score 0-100 from the YOLO-measured change plus other indicators. Returns (score, notes).
    change 40 + size 20 + long-term inventory growth 20 + lake type 20 (Low <30, Medium <60, High)."""
    flags = []
    if after_ha <= 0:
        change, size_ha = 0, fallback_ha
        flags.append("YOLO found no lake in the after image: change not measured, size from satellite water detection")
    elif before_ha <= 0:
        change, size_ha = 30 * clamp01(after_ha / 5), after_ha
        flags.append("NEW: YOLO found no lake in the before image")
    else:
        gain = after_ha - before_ha
        change, size_ha = 40 * clamp01(gain / before_ha / 0.5) * clamp01(gain / 5), after_ha
        if after_ha < 0.75 * before_ha:
            flags.append(f"area fell to {after_ha / before_ha * 100:.0f}% of the before image: "
                         "partial detection or real drainage, verify")
    size = 20 * clamp01(size_ha / 100)
    if series:
        a2020, first = series[2020]["area_ha"], min(series)
        if first < 2020:
            a0 = series[first]["area_ha"]
            hist = 20 * clamp01((a2020 / a0 - 1) if a0 else 0) * clamp01((a2020 - a0) / 10)
        else:
            hist = 10
            flags.append("only in 2020 inventory")
        kind = type_points(series[2020]["type"])
    else:
        hist, kind = 10, 10
        flags.append("not in 2020 inventory")
    return change + size + hist + kind, flags


def region_report(region, lakes, places, years):
    """Plain-text summary of every lake, built after ALL lakes are analysed."""
    ok = sorted((l for l in lakes if l.get("risk_score") is not None),
                key=lambda l: l["risk_score"], reverse=True)
    failed = [l for l in lakes if l.get("risk_score") is None]
    n = {lvl: sum(1 for l in ok if l["risk_label"] == lvl) for lvl in LEVEL_RANK}
    lines = [f"GLACIAL LAKE SCREENING REPORT - {region}",
             "Automated satellite screening, not a confirmed flood warning. Please verify with field data.",
             f"Images compared: {years[0]} vs {years[1]} (Landsat). Lake areas measured with YOLO.",
             f"Lakes analysed: {len(ok)} | High: {n['High']} | Medium: {n['Medium']} | Low: {n['Low']}"
             + (f" | not analysed: {len(failed)}" if failed else ""), ""]
    if not lakes:
        lines.append("No lakes were found.")
    for l in ok:
        change = f"{l['change_pct']:+.0f}%" if l["change_pct"] is not None else "new lake"
        lines.append(f"{l['name']} ({l['lat']}, {l['lon']}): {l['risk_label']} (score {l['risk_score']}/100)")
        lines.append(f"   Area {years[0]} -> {years[1]}: {l['before_ha']} ha -> {l['after_ha']} ha ({change})")
        if l["history_ha"]:
            lines.append(f"   Inventory history (ha): {l['history_ha']}")
        if l["notes"]:
            lines.append(f"   Notes: {l['notes']}")
        pl = places.get(l["key"], [])
        if pl:
            lines.append("   Downstream: " + ", ".join(f"{p['place']} ({p['river_km']} km)" for p in pl[:MAX_PLACES]))
        lines.append("")
    for l in failed:
        lines.append(f"{l['name']}: not analysed ({l['error']})")
    return "\n".join(lines).strip()


def region_alerts(region, lakes, places, years):
    """Returns (summary report text, planned emails).
    Authorities (no place, the region name, or a downstream place) get the full report when any lake
    is Medium or High. Residents of a downstream place get a short notice when a High lake is above them."""
    report = region_report(region, lakes, places, years)
    ok = sorted((l for l in lakes if l.get("risk_score") is not None),
                key=lambda l: l["risk_score"], reverse=True)
    level = max((l["risk_label"] for l in ok), key=LEVEL_RANK.get, default="Low")
    if LEVEL_RANK[level] < 1:
        return report, []
    downstream = {p["place"].lower() for pl in places.values() for p in pl}
    region_key = f"region:{region.lower()}"
    alerts = []
    for c in (c for c in db.list_contacts() if c["active"]):
        cplace = c["place"].lower()
        if c["role"] == "authority" and (not cplace or cplace == region.lower() or cplace in downstream):
            subject = f"[{level.upper()}] Glacial lake screening report - {region}"
            body = report
        elif c["role"] == "resident" and level == "High" and cplace in downstream:
            for lake in (l for l in ok if l["risk_label"] == "High"):
                match = {p["place"].lower(): p for p in places.get(lake["key"], [])}.get(cplace)
                if match:
                    subject, body = resident_message(lake, match)
                    break
            else:
                continue
        else:
            continue
        alerts.append({"lake_id": "all", "lake_key": region_key, "level": level,
                       "to_name": c["name"], "to_email": c["email"], "role": c["role"],
                       "subject": subject, "body": body, "status": "planned"})
    return report, alerts
