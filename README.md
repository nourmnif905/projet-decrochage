# Prédire le décrochage étudiant : application de classification explicable

Projet Challenge 1 · Data Mining · II3-IA / II3-IF · ENSI · 2026-2027
Cas n°10 du catalogue : *Predict Students' Dropout and Academic Success* (UCI, id=697).

## Fonctionnalités (état actuel)
| Module | Page | État |
|---|---|---|
| 1. Données | `1_Donnees` | ✅ |
| 2. Préparation | `2_Preparation` | ✅ |
| 3. Modélisation | `3_Modelisation` | ✅ |
| 4. Évaluation | `4_Evaluation` | ✅ |
| 5. Explicabilité (SHAP, LIME) | | à venir |
| 6. API FastAPI | `api/main.py` + page `6_Prediction_API` | ✅ prédiction (routes SHAP / LIME à venir) |

## Lancer l'application
```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux : source .venv/bin/activate
pip install -r requirements.txt -r api/requirements.txt
```
Deux terminaux (venv activé, depuis la racine du projet) :
```bash
uvicorn api.main:app --port 8000     # terminal 1 : l'API  (docs : http://127.0.0.1:8000/docs)
streamlit run app/main.py            # terminal 2 : l'application
```
L'API lit les modèles dans `models/` (créés par `modelisation.ipynb` et `evaluation_finale.ipynb`).
L'adresse de l'API se règle avec la variable d'environnement `API_URL` (défaut : `http://127.0.0.1:8000`).

## API
| Route | Rôle |
|---|---|
| `GET /health` | État du service |
| `GET /models` | Modèles disponibles, seuils, métriques |
| `GET /features?model_id=...` | Variables attendues (types, bornes, valeurs par défaut) |
| `POST /predict` | Probabilité de décrochage et décision pour un étudiant |
| `POST /predict/batch` | Plusieurs étudiants à la fois |
| `POST /reload` | Recharger les modèles après un nouvel entraînement |

Tests : `pytest tests/`
Les données sont téléchargées automatiquement (UCI). Une copie locale `data/students_dropout.csv` sert de plan B.

## Structure
```
app/            main.py (routeur), utils.py, pages/
api/            main.py (FastAPI), requirements.txt
tests/          test_api.py
models/         modèles sauvegardés par les notebooks
data/           students_dropout.csv (copie locale)
notebooks/      exploration.ipynb, modelisation.ipynb, evaluation_finale.ipynb
requirements.txt
```

## Liens et captures d'écran
- Application déployée : _à compléter_
- API : _à compléter_
- Captures d'écran : _à ajouter dans `docs/`_
