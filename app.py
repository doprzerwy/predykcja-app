import streamlit as st
import json
import re
import csv
import io

from sklearn.ensemble import RandomForestClassifier

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ========================
# GOOGLE DRIVE
# ========================

SERVICE_ACCOUNT_INFO = dict(
    st.secrets["gcp_service_account"]
)

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

credentials = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO,
    scopes=SCOPES
)

service = build(
    "drive",
    "v3",
    credentials=credentials
)

# ========================
# STREAMLIT UI
# ========================

st.title("Predykcje doprzerwy.pl")

LIGA = st.selectbox(
    "Liga",
    [
        "IV liga",
        "KO Wrocław",
        "KO JG",
        "KO Wałbrzych",
        "KO Legnica"
    ]
)

KOLEJKA = st.text_input(
    "Kolejka",
    "26"
)

# ========================
# GOOGLE DRIVE FILE IDS
# ========================

files_map = {

    "IV liga": (
        "1VMQmHFc4YfQ_ALGimkMiw8510XmKwBSr",
        "1eqeTd9OGbCBOtplUcOi2bwmOLUMNxb31"
    ),

    "KO Wrocław": (
        "1E_E2cLvcUKbN5ZyiwoGYa6k6EWyeQCt0",
        "TUTAJ_WSTAW_ID_2026_2027"
    ),

    "KO JG": (
        "154Vr3dq03u0Q2ufRq508R15ovzfAkvuM",
        "TUTAJ_WSTAW_ID_2026_2027"
    ),

    "KO Wałbrzych": (
        "1K48Tucl4H_ibqvQMKVK8ZK-zB5GspGMY",
        "1JdxUs4qHBaropvpKEfJK2VUQAlR_SWyh"
    ),

    "KO Legnica": (
        "1D78lfbIfFiGuhgtNdQ9VOOYTwln_3ZXa",
        "TUTAJ_WSTAW_ID_2026_2027"
    )
}

# ========================
# HELPERS
# ========================

def avg(x):

    return sum(x) / len(x) if x else 0

def get_last(x, n=5):

    return x[-n:]

def clean_team(x):

    x = re.sub(
        r'\d{1,2} \w+, \d{2}:\d{2}',
        '',
        x
    )

    return x.replace(
        "–",
        "-"
    ).strip().lower()

def find_team(name, teams):

    name = name.lower().strip()

    for t in teams:

        if name in t.lower():

            return t

    return None

# ========================
# DOWNLOAD JSON FROM DRIVE
# ========================

@st.cache_data(ttl=300)
def load_json_from_drive(file_id):

    request = service.files().get_media(
        fileId=file_id
    )

    file = io.BytesIO()

    downloader = MediaIoBaseDownload(
        file,
        request
    )

    done = False

    while done is False:

        status, done = downloader.next_chunk()

    file.seek(0)

    return json.load(
        io.TextIOWrapper(
            file,
            encoding="utf-8"
        )
    )

# ========================
# LOAD DATA
# ========================

f1, f2 = files_map[LIGA]

data_2025 = load_json_from_drive(f1)
data_2026 = load_json_from_drive(f2)

data = {
    "kolejki":
        data_2025["kolejki"] +
        data_2026["kolejki"]
}

# ========================
# BUILD TEAMS + MATCHES
# ========================

teams = {}
all_matches = []

for k in data["kolejki"]:

    for m in k["mecze"]:

        h = m["home"]
        a = m["away"]

        g1 = m["score"]["home"]
        g2 = m["score"]["away"]

        for t in [h, a]:

            teams.setdefault(
                t,
                {
                    "points": [],
                    "scored": [],
                    "conceded": []
                }
            )

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
            "score": {
                "home": g1,
                "away": g2
            }
        })

# ========================
# MODEL
# ========================

X = []
y = []

for k in data["kolejki"]:

    for m in k["mecze"]:

        h = m["home"]
        a = m["away"]

        hd = teams[h]
        ad = teams[a]

        feats = [

            sum(get_last(hd["points"])) -
            sum(get_last(ad["points"])),

            (
                avg(get_last(hd["scored"])) -
                avg(get_last(hd["conceded"]))
            ) -
            (
                avg(get_last(ad["scored"])) -
                avg(get_last(ad["conceded"]))
            ),

            avg(get_last(hd["scored"])) -
            avg(get_last(ad["conceded"])),

            avg(get_last(ad["scored"])) -
            avg(get_last(hd["conceded"]))
        ]

        X.append(feats)

        y.append(
            m["result"]
        )

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ========================
# H2H
# ========================

def get_h2h(home, away):

    res = []

    for m in all_matches:

        if (
            (
                m["home"] == home and
                m["away"] == away
            ) or
            (
                m["home"] == away and
                m["away"] == home
            )
        ):

            res.append(m)

    if not res:

        return []

    last = res[-1]

    return [
        f"{last['home']} "
        f"{last['score']['home']}-"
        f"{last['score']['away']} "
        f"{last['away']}"
    ]

# ========================
# INPUT
# ========================

future_text = st.text_area(
    "Wklej mecze (A - B)",
    height=200
)

# ========================
# CONFIDENCE
# ========================

def level(r):

    c = max(
        r["p1"],
        r["px"],
        r["p2"]
    )

    if c > 0.60:
        return "🟢"

    elif c > 0.52:
        return "🟡"

    else:
        return "🔴"

# ========================
# GENERATE
# ========================

if st.button("Generuj"):

    lines = future_text.split("\n")

    matches = []

    for line in lines:

        if "-" in line:

            parts = line.split("-")

            matches.append({

                "home_raw":
                    parts[0].strip(),

                "away_raw":
                    parts[1].strip()
            })

    results = []

    for m in matches:

        h_raw = m["home_raw"]
        a_raw = m["away_raw"]

        h_key = find_team(
            h_raw,
            teams
        )

        a_key = find_team(
            a_raw,
            teams
        )

        st.write(
            "DEBUG:",
            h_raw,
            "->",
            h_key
        )

        st.write(
            "DEBUG:",
            a_raw,
            "->",
            a_key
        )

        if not h_key or not a_key:

            continue

        hd = teams[h_key]
        ad = teams[a_key]

        feats = [[

            sum(get_last(hd["points"])) -
            sum(get_last(ad["points"])),

            (
                avg(get_last(hd["scored"])) -
                avg(get_last(hd["conceded"]))
            ) -
            (
                avg(get_last(ad["scored"])) -
                avg(get_last(ad["conceded"]))
            ),

            avg(get_last(hd["scored"])) -
            avg(get_last(ad["conceded"])),

            avg(get_last(ad["scored"])) -
            avg(get_last(hd["conceded"]))
        ]]

        pred = model.predict(
            feats
        )[0]

        prob = model.predict_proba(
            feats
        )[0]

        results.append({

            "home": h_key,
            "away": a_key,

            "prediction": pred,

            "p1": round(prob[0], 2),
            "px": round(prob[1], 2),
            "p2": round(prob[2], 2),

            "h2h": get_h2h(
                h_key,
                a_key
            )
        })

    # ====================
    # OUTPUT
    # ====================

    out = "TYPY:\n\n"

    for r in results:

        out += (
            f"{r['home']} – "
            f"{r['away']} → "
            f"{r['prediction']}\n"
        )

    out += "\n=== DANE DO ANALIZY ===\n\n"

    for r in results:

        out += (
            f"{r['home']} vs "
            f"{r['away']}\n"
        )

        out += (
            f"Typ: "
            f"{r['prediction']}\n"
        )

        out += (
            f"1: {r['p1']} "
            f"X: {r['px']} "
            f"2: {r['p2']}\n"
        )

        out += (
            f"H2H: "
            f"{r['h2h'][-1] if r['h2h'] else 'brak'}\n"
        )

        out += "---\n"

    st.code(out)

    # ====================
    # CSV
    # ====================

    max_matches = 9

    headers = [
        "Liga",
        "Kolejka"
    ]

    row = [
        LIGA,
        KOLEJKA
    ]

    for i in range(1, max_matches + 1):

        headers += [
            f"home{i}",
            f"away{i}",
            f"typ{i}",
            f"lvl{i}"
        ]

    for i in range(max_matches):

        if i < len(results):

            r = results[i]

            row += [
                r["home"],
                r["away"],
                r["prediction"],
                level(r)
            ]

        else:

            row += [
                "",
                "",
                "",
                ""
            ]

    if results:

        best = max(
            results,
            key=lambda r:
                max(
                    r["p1"],
                    r["px"],
                    r["p2"]
                )
        )

        pewniak = (
            f"{best['home']} - "
            f"{best['away']} "
            f"({best['prediction']}) "
            f"{level(best)}"
        )

    else:

        pewniak = ""

    headers.append("pewniak")

    row.append(pewniak)

    buffer = io.StringIO()

    writer = csv.writer(buffer)

    writer.writerow(headers)
    writer.writerow(row)

    st.download_button(
        "Pobierz canva.csv",
        buffer.getvalue(),
        file_name="canva.csv"
    )
