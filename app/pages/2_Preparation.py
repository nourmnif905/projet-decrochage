import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from utils import build_prep, detect_types, ensure_data

st.title("🧹 Module 2 · Préparation")
ensure_data()

df, target = st.session_state["df"], st.session_state["target"]
all_feats = [c for c in df.columns if c != target]
st.write(f"**Variable cible :** `{target}` (1 = classe d'intérêt)")

# ------------------------------------------------------------------ variables
st.subheader("1. Variables explicatives")
has_sem = any("sem" in c for c in all_feats)
scenario = "Toutes les variables"
if has_sem:
    scenario = st.radio(
        "Scénario (jeu décrochage)",
        ["Toutes les variables", "Sans notes du 2e semestre", "Sans 1er ni 2e semestre (détection précoce)"],
        help="Plus on retire de résultats scolaires, plus la détection est précoce, mais moins précise.")
default = all_feats
if scenario == "Sans notes du 2e semestre":
    default = [c for c in all_feats if "2nd sem" not in c]
elif scenario.startswith("Sans 1er"):
    default = [c for c in all_feats if "1st sem" not in c and "2nd sem" not in c]

features = st.multiselect("Variables à utiliser", all_feats, default=default,
                          key=f"feats_{scenario}")
if not features:
    st.error("Choisis au moins une variable.")
    st.stop()

# ------------------------------------------------------------------ types
st.subheader("2. Types de variables")
auto_cat, auto_bin, auto_num = detect_types(df, features)
cat = st.multiselect("Variables catégorielles (encodées en one-hot)", features,
                     default=auto_cat, key=f"cat_{scenario}",
                     help="Les codes (formation, profession...) ne sont pas des quantités : "
                          "il faut les traiter comme des catégories.")
rest = [c for c in features if c not in cat]
binary = [c for c in rest if df[c].nunique() == 2]
num = [c for c in rest if c not in binary]
st.caption(f"{len(cat)} catégorielles · {len(binary)} binaires · {len(num)} numériques")

# ------------------------------------------------------------------ nettoyage
st.subheader("3. Valeurs manquantes, encodage, normalisation")
c1, c2, c3 = st.columns(3)
na_mode = c1.selectbox("Valeurs manquantes", ["Imputer (médiane)", "Imputer (moyenne)", "Supprimer les lignes"])
scaling = c2.selectbox("Normalisation des numériques",
                       ["Standardisation (StandardScaler)", "Min-Max (0 à 1)", "Aucune"])
min_freq = c3.slider("Regrouper les modalités rares (effectif minimum)", 1, 100, 20,
                     help="Évite de créer des centaines de colonnes pour des codes très rares.")
n_na = int(df[features].isna().sum().sum())
st.caption(f"{n_na} valeur(s) manquante(s) dans les variables choisies.")

# ------------------------------------------------------------------ split
st.subheader("4. Séparation entraînement / test")
c1, c2, c3 = st.columns(3)
test_pct = c1.slider("Part du jeu de test (%)", 10, 40, 20, step=5)
seed = c2.number_input("Graine aléatoire", 0, 9999, 42)
stratify = c3.checkbox("Stratifier (garder la proportion des classes)", True)

if st.button("✅ Valider la préparation", type="primary"):
    data = df[features + [target]]
    if na_mode == "Supprimer les lignes":
        data = data.dropna()
    X, y = data[features], data[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_pct / 100, random_state=int(seed),
        stratify=y if stratify else None)
    cfg = {"cat": cat, "bin": binary, "num": num, "scaling": scaling, "min_freq": min_freq,
           "impute": "median" if na_mode != "Imputer (moyenne)" else "mean"}
    n_enc = build_prep(cfg, features).fit(X_tr).transform(X_tr).shape[1]
    st.session_state.update(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                            prep_cfg=cfg, features=features, n_encoded=n_enc, seed=int(seed))
    for k in ["fitted", "probas", "table"]:
        st.session_state.pop(k, None)           # la préparation a changé : modèles à refaire

if "X_train" in st.session_state:
    s = st.session_state
    st.success("Préparation validée.")
    a, b, c, d = st.columns(4)
    a.metric("Train", f"{len(s['X_train'])} lignes")
    b.metric("Test", f"{len(s['X_test'])} lignes")
    c.metric("Positifs (train / test)", f"{s['y_train'].mean():.1%} / {s['y_test'].mean():.1%}")
    d.metric("Colonnes après encodage", s["n_encoded"])
    st.info("Le préprocesseur est ajusté **uniquement sur le train** (dans un pipeline) : "
            "aucune information du test ne fuit vers l'apprentissage.")
    st.write("Étape suivante : page **3 · Modélisation**.")
