import streamlit as st
import json, re, csv
from sklearn.ensemble import RandomForestClassifier

# ========================
# KONFIG
# ========================
BASE_PATH = "predykcja_dane/"

LIGA = st.selectbox("Liga", ["IV liga", "KO Wrocław", "KO JG", "KO Wałbrzych", "KO Legnica"])
KOLEJKA = st.text_input("Kolejka", "26")

files_map = {
    "IV liga": ("iv_liga_2025_2026.json", "iv_liga_2026_2027.json"),
    "KO Wrocław": ("ko_wroclaw_2025_2026.json", "ko_wroclaw_2026_2027.json"),
    "KO JG": ("ko_jg_2025_2026.json", "ko_jg_2026_2027.json"),
    "KO Wałbrzych": ("ko_walbrzych_2025_2026.json", "ko_walbrzych_2026_2027.json"),
    "KO Legnica": ("ko_legnica_2025_2026.json", "ko_legnica_2026_2027.json"),
}

# ========================
# FUNKCJE
# ========================
def avg(x): return sum(x)/len(x) if x else 0
def get_last(x, n=5): return x[-n:]

def clean_team(x):
    x = re.sub(r'\d{1,2} \w+, \d{2}:\d{2}', '', x)
    return x.replace("–", "-").strip().lower()

def find_team(name, teams):
    name = name.lower().strip()
    for t in teams:
        if name in t.lower():
            return t
    return None

# ========================
# LOAD DATA
# ========================
f1, f2 = files_map[LIGA]

with open(BASE_PATH + f1, encoding="utf-8") as a, open(BASE_PATH + f2, encoding="utf-8") as b:
    data = {"kolejki": json.load(a)["kolejki"] + json.load(b)["kolejki"]}

# ========================
# BUILD TEAMS + MATCHES
# ========================
teams = {}
all_matches = []

for k in data["kolejki"]:
    for m in k["mecze"]:
        h, a = m["home"], m["away"]
        g1, g2 = m["score"]["home"], m["score"]["away"]

        for t in [h, a]:
            teams.setdefault(t, {"points": [], "scored": [], "conceded": []})

        if g1 > g2:
            teams[h]["points"].append(3)
            teams[a]["points"].append(0)
        elif g1 < g2:
            teams[h]["points"].append(0)
            teams[a]["points"].append(3)
        else:
            teams[h]["points"].append(1)
            teams[a]["points"].append(1)

        teams[h]["scored"].append(g1)
        teams[h]["conceded"].append(g2)
        teams[a]["scored"].append(g2)
        teams[a]["conceded"].append(g1)

        all_matches.append({
            "home": h,
            "away": a,
            "score": {"home": g1, "away": g2}
        })

# ========================
# MODEL
# ========================
X, y = [], []

for k in data["kolejki"]:
    for m in k["mecze"]:
        h, a = m["home"], m["away"]
        hd, ad = teams[h], teams[a]

        feats = [
            sum(get_last(hd["points"])) - sum(get_last(ad["points"])),
            (avg(get_last(hd["scored"])) - avg(get_last(hd["conceded"]))) -
            (avg(get_last(ad["scored"])) - avg(get_last(ad["conceded"]))),
            avg(get_last(hd["scored"])) - avg(get_last(ad["conceded"])),
            avg(get_last(ad["scored"])) - avg(get_last(hd["conceded"]))
        ]

        X.append(feats)
        y.append(m["result"])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ========================
# H2H
# ========================
def get_h2h(home, away):
    res = []

    for m in all_matches:
        if (
            (m["home"] == home and m["away"] == away) or
            (m["home"] == away and m["away"] == home)
        ):
            res.append(m)

    if not res:
        return []

    last = res[-1]
    return [f"{last['home']} {last['score']['home']}-{last['score']['away']} {last['away']}"]

# ========================
# UI
# ========================
future_text = st.text_area("Wklej mecze (A - B)", height=200)

def level(r):
    c = max(r["p1"], r["px"], r["p2"])
    return "🟢" if c > 0.60 else "🟡" if c > 0.52 else "🔴"

if st.button("Generuj"):

    lines = future_text.split("\n")
    matches = []

    for line in lines:
        if "-" in line:
            parts = line.split("-")
            matches.append({
                "home_raw": parts[0].strip(),
                "away_raw": parts[1].strip()
            })

    results = []

    for m in matches:
        h_raw = m["home_raw"]
        a_raw = m["away_raw"]

        h_key = find_team(h_raw, teams)
        a_key = find_team(a_raw, teams)

        # DEBUG (możesz usunąć później)
        st.write("DEBUG:", h_raw, "->", h_key)
        st.write("DEBUG:", a_raw, "->", a_key)

        if not h_key or not a_key:
            continue

        hd = teams[h_key]
        ad = teams[a_key]

        feats = [[
            sum(get_last(hd["points"])) - sum(get_last(ad["points"])),
            (avg(get_last(hd["scored"])) - avg(get_last(hd["conceded"]))) -
            (avg(get_last(ad["scored"])) - avg(get_last(ad["conceded"]))),
            avg(get_last(hd["scored"])) - avg(get_last(ad["conceded"])),
            avg(get_last(ad["scored"])) - avg(get_last(hd["conceded"]))
        ]]

        pred = model.predict(feats)[0]
        prob = model.predict_proba(feats)[0]

        results.append({
            "home": h_key,
            "away": a_key,
            "prediction": pred,
            "p1": round(prob[0], 2),
            "px": round(prob[1], 2),
            "p2": round(prob[2], 2),
            "h2h": get_h2h(h_key, a_key)
        })

    # ========================
    # OUTPUT
    # ========================
    out = "TYPY:\n\n"

    for r in results:
        out += f"{r['home']} – {r['away']} → {r['prediction']}\n"

    out += "\n=== DANE DO ANALIZY ===\n\n"

    for r in results:
        out += f"{r['home']} vs {r['away']}\n"
        out += f"Typ: {r['prediction']}\n"
        out += f"1: {r['p1']} X: {r['px']} 2: {r['p2']}\n"
        out += f"H2H: {r['h2h'][-1] if r['h2h'] else 'brak'}\n"
        out += "---\n"

    st.code(out)

    # ========================
    # CSV
    # ========================
    max_matches = 9

    headers = ["Liga", "Kolejka"]
    row = [LIGA, KOLEJKA]

    for i in range(1, max_matches + 1):
        headers += [f"home{i}", f"away{i}", f"typ{i}", f"lvl{i}"]

    for i in range(max_matches):
        if i < len(results):
            r = results[i]
            row += [r["home"], r["away"], r["prediction"], level(r)]
        else:
            row += ["", "", "", ""]

    if results:
        best = max(results, key=lambda r: max(r["p1"], r["px"], r["p2"]))
        pewniak = f"{best['home']} - {best['away']} ({best['prediction']}) {level(best)}"
    else:
        pewniak = ""

    headers.append("pewniak")
    row.append(pewniak)

    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerow(row)

    st.download_button("Pobierz canva.csv", buffer.getvalue(), file_name="canva.csv")
