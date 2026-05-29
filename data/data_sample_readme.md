# SpiGes Synthetic Sample Dataset

> **GovTech Hackathon 2026 · Swiss ED Predictor**  
> Données synthétiques générées à partir du schéma officiel SpiGes v1.4/1.5  
> Source schema : [I14Y register.ld.admin.ch — SpiGes_Administratives TTL](https://register.ld.admin.ch/i14y/dataset/SpiGes_Administratives)

---

## ⚠️ Important

Ces données sont **entièrement synthétiques** — générées algorithmiquement pour reproduire les patterns statistiques réels des urgences suisses. Elles ne contiennent **aucune donnée patient réelle**. Elles sont conformes à la structure officielle SpiGes et peuvent être remplacées par les vraies données OFS/OFSP le moment venu.

---

## 📁 Fichiers

| Fichier | Lignes | Colonnes | Usage |
|---------|--------|----------|-------|
| `spiges_daily_aggregated.csv` | 2 479 | 26 | **Principal — entraînement ML** |
| `spiges_synthetic_patients.csv` | 50 000 | 42 | Données patient-niveau — exploration |

---

## 🏥 Hôpitaux couverts

| BUR-Nr | Canton | Hôpital |
|--------|--------|---------|
| 301001 | BE | Inselspital Bern |
| 351001 | ZH | USZ Zürich |
| 630001 | GE | HUG Genève |
| 220001 | VD | CHUV Lausanne |
| 120001 | BS | USB Basel |
| 190001 | AG | KSA Aarau |
| 170001 | SG | KSSG St. Gallen |
| 230001 | VS | Hôpital du Valais |
| 240001 | NE | HNE Neuchâtel |
| 290001 | TI | EOC Ticino |

**Période couverte :** 2021-01-01 → 2024-09-05

---

## 📋 `spiges_daily_aggregated.csv` — Dictionnaire des colonnes

Ce fichier est l'agrégation quotidienne par hôpital. C'est le **fichier d'entrée direct pour XGBoost**.

### Identifiants

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | date | Date du jour (YYYY-MM-DD) |
| `burnr_gesv` | string | Numéro BUR de l'hôpital (9 chiffres, registre OFS) |
| `kanton_hospital` | string | Code canton de l'hôpital (BE, ZH, GE...) |
| `hospital_name` | string | Nom lisible de l'hôpital (hors SpiGes officiel) |

---

### 🎯 Variables cibles (à prédire)

| Colonne | Type | Description |
|---------|------|-------------|
| `notfall_admissions` | int | **Nombre d'arrivées aux urgences ce jour-là** — variable cible principale |
| `target_notfall_next24h` | int | Admissions urgences le lendemain (J+1) |
| `target_notfall_next48h` | int | Admissions urgences dans 2 jours (J+2) |
| `target_notfall_next72h` | int | Admissions urgences dans 3 jours (J+3) |

> **Usage :** entraîner 3 modèles séparés selon l'horizon de prédiction souhaité.  
> Les lignes des derniers 3 jours auront des NaN sur ces colonnes — à exclure du training.

---

### 📅 Features temporelles

| Colonne | Type | Valeurs | Description |
|---------|------|---------|-------------|
| `year` | int | 2021–2024 | Année |
| `month` | int | 1–12 | Mois (1=janvier) |
| `day_of_week` | int | 0–6 | Jour de la semaine (0=lundi, 6=dimanche) |
| `week_of_year` | int | 1–53 | Numéro de semaine ISO |
| `is_weekend` | int | 0/1 | 1 si samedi ou dimanche |
| `is_winter` | int | 0/1 | 1 si décembre, janvier ou février |
| `is_summer` | int | 0/1 | 1 si juin, juillet ou août |

> **Corrélations observées :** `is_winter` → +0.286, `is_weekend` → -0.276 avec `notfall_admissions`

---

### 🔁 Features de lag (mémoire historique)

| Colonne | Type | Description | Corrélation |
|---------|------|-------------|-------------|
| `notfall_lag1` | float | Admissions urgences **hier** (J-1) | **0.767** |
| `notfall_lag7` | float | Admissions urgences **il y a 7 jours** (même jour semaine dernière) | **0.859** |
| `notfall_roll7` | float | Moyenne glissante des **7 derniers jours** (tendance locale) | ~0.75 |

> Ce sont les features les plus prédictives. Un mercredi ressemble toujours à un mercredi.  
> Les 7 premières lignes par hôpital auront des NaN sur lag7 — comportement normal.

---

### 👥 Features démographiques et cliniques (agrégées)

| Colonne | Type | Valeurs | Description |
|---------|------|---------|-------------|
| `total_admissions` | int | — | Total admissions du jour (urgences + planifiées) |
| `mean_age` | float | 0–100 | Âge moyen des patients admis |
| `pct_elderly` | float | 0–1 | Part des patients ≥ 65 ans (proxy risque hivernal) |
| `mean_severity` | float | 1–4 | Score de sévérité moyen (`schwere_score` SpiGes) |
| `mean_nems` | float | 0–60 | Score NEMS moyen (charge en soins infirmiers) |
| `ips_cases` | int | — | Nombre de cas transférés en soins intensifs (IPS) |
| `mean_los_hours` | float | — | Durée de séjour moyenne en heures |

> `pct_elderly` → corrélation 0.189 avec volume urgences.  
> En hiver, cette valeur monte significativement (vagues respiratoires, chutes).

---

## 📋 `spiges_synthetic_patients.csv` — Dictionnaire des colonnes

Ce fichier contient une ligne par patient. Utile pour l'exploration et la compréhension des patterns, ou pour construire des features avancées.

### Identifiants et dates

| Colonne | Type | Description |
|---------|------|-------------|
| `burnr_gesv` | string | Numéro BUR de l'hôpital (champ SpiGes officiel) |
| `kanton_hospital` | string | Canton de l'hôpital |
| `hospital_name` | string | Nom lisible |
| `eintrittsdatum` | string | Date d'admission format YYYYMMDD (SpiGes officiel) |
| `austrittsdatum` | string | Date de sortie format YYYYMMDD (SpiGes officiel) |
| `year` / `month` / `day_of_week` | int | Composantes temporelles parsées |
| `is_weekend` / `is_winter` / `is_summer` | int | Flags temporels |

---

### Champs SpiGes officiels — Administratif

| Colonne | Type | Valeurs | Description (SpiGes) |
|---------|------|---------|----------------------|
| `abc_fall` | string | A / B / C | Type de cas statistique : A=sortie dans l'année, B=à cheval sur 2 ans, C=présent toute l'année |
| `eintrittsart` | int | 1–9 | Mode d'admission : **1=Urgence**, 2=Planifié, 3=Transfert, 4=Naissance, 9=Autre |
| `is_notfall` | int | 0/1 | Raccourci : 1 si `eintrittsart == 1` (urgence) |
| `hauptleistungsstelle` | int | 1–9 | Service principal : **1=Urgences**, 2=Chirurgie, 3=Médecine interne, 4=Gynéco, 5=Pédiatrie, 6=Psychiatrie, 7=Neurologie, 8=Cardiologie, 9=Autre |
| `einw_instanz` | int | 1–5 | Entité référente : 1=Auto-référé, 2=Médecin de famille, 3=Spécialiste, 4=Services d'urgence (144), 5=Autre hôpital |
| `geschlecht` | int | 1/2 | Sexe : 1=Homme, 2=Femme |
| `alter` | int | 0–99 | Âge en années révolues |
| `alter_group` | string | — | Classe d'âge (0-14, 15-29, 30-44, 45-64, 65-79, 80+) |
| `wohnkanton` | string | BE, ZH... | Canton de domicile du patient |
| `wohnland` | string | CH, DE... | Pays de domicile |
| `nationalitaet` | string | CH, DE... | Nationalité |
| `versicherungsklasse` | int | 1–3 | Classe d'assurance : 1=Générale, 2=Semi-privée, 3=Privée |
| `grundversicherung` | int | 1–3 | Assurance de base : 1=LAMal, 2=Accident, 3=Autre |
| `liegeklasse` | int | 1–3 | Classe d'hospitalisation : 1=Commune, 2=Semi-privée, 3=Privée |
| `tarif` | string | DRG / TARPSY / ST_Reha | Système tarifaire appliqué |
| `sekundaertransport` | int | 0/1 | 1 = transfert secondaire depuis un autre hôpital |

---

### Champs SpiGes officiels — Cliniques / Scores

| Colonne | Type | Valeurs | Description (SpiGes) |
|---------|------|---------|----------------------|
| `schwere_score` | int | 1–4 | Score de sévérité du cas : 1=léger, 4=critique |
| `nems` | int | 0–60 | Score NEMS (Nine Equivalents of Nursing Manpower) — charge infirmière |
| `art_score` | int | 0–4 | Score ART (Acute Risk of Transfert) |
| `aufenthalt_ips` | int | 0–N | Jours en soins intensifs (IPS) — 0 si aucun |
| `aufenthalt_imc` | int | 0–N | Jours en soins intermédiaires (IMC) |
| `aufwand_imc` | int | 0–N | Heures de soins intermédiaires |
| `beatmung` | int | 0–N | Heures de ventilation mécanique — 0 si aucune |
| `chlz` | int | — | Cost Weight (pondération des coûts DRG) |

---

### Champs SpiGes officiels — Sortie

| Colonne | Type | Valeurs | Description (SpiGes) |
|---------|------|---------|----------------------|
| `austrittsentscheid` | int | 1–9 | Décision de sortie : 1=Guéri, 2=Amélioré, 3=Inchangé, 4=Détérioré, 5=Décédé, 9=Autre |
| `austritt_aufenthalt` | int | 0–99 | Type de séjour post-sortie : 0=domicile, 1=EMS, 2=rééducation... |
| `austritt_behandlung` | int | 0–9 | Traitement post-sortie : 0=aucun, 1=ambulatoire, 2=réhabilitation... |
| `los_hours` | int | — | Durée de séjour en heures (calculé) |
| `los_days` | int | — | Durée de séjour en jours (calculé) |
| `admin_urlaub` | int | 0–N | Heures de congé administratif (patient hors établissement avec lit réservé) |

---

## 🚀 Quickstart Python

```python
import pandas as pd

# Fichier principal pour ML
daily = pd.read_csv("data/sample/spiges_daily_aggregated.csv", parse_dates=["date"])

# Features de base
features = [
    "month", "day_of_week", "is_weekend", "is_winter", "is_summer",
    "pct_elderly", "mean_severity",
    "notfall_lag1", "notfall_lag7", "notfall_roll7"
]

# Prédire J+1
df_train = daily.dropna(subset=["target_notfall_next24h"])
X = df_train[features]
y = df_train["target_notfall_next24h"]

# Exploration patient-niveau
patients = pd.read_csv("data/sample/spiges_synthetic_patients.csv")
print(patients.groupby(["is_winter", "is_notfall"]).size())
```

---

## 📊 Statistiques descriptives clés

| Métrique | Valeur |
|----------|--------|
| Taux d'urgences (`is_notfall`) | 57.9% |
| Admissions ED moyennes / jour / hôpital | 11.7 |
| Admissions ED hiver vs été | 15.1 vs 9.4 (+61%) |
| Part patients ≥ 65 ans | ~22% (hiver : ~38%) |
| Feature la plus corrélée | `notfall_lag7` (r=0.859) |

---

## 🔗 Références

- [Schéma SpiGes TTL officiel — I14Y](https://register.ld.admin.ch/i14y/dataset/SpiGes_Administratives)
- [Exemple patients.xml — GovTech Hackathon 2026](https://github.com/I14Y-ch/govtech-hackathon-2026/blob/main/data/patients.xml)
- [Documentation SpiGes — OFSP](https://www.bag.admin.ch/spiges)
- [Obsan — Taux de recours aux urgences](https://ind.obsan.admin.ch/fr/indicator/obsan/taux-de-recours-aux-services-durgence)

---

*Généré pour le GovTech Hackathon 2026 · Swiss ED Predictor · Licence MIT*