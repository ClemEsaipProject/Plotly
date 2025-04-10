# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import statsmodels.api as sm
import nbformat


# Fonction de prétraitement pour le fichier us_covid19_daily.csv
def preprocess_us_covid19_daily():
    file_path = "data/covid_csv/us_covid19_daily.csv"
    df = pd.read_csv(file_path)
    
    # Conversion des colonnes numériques
    numeric_cols = ['positive', 'negative', 'pending', 'hospitalizedCurrently', 
                    'hospitalizedCumulative', 'inIcuCurrently', 'inIcuCumulative', 
                    'onVentilatorCurrently', 'onVentilatorCumulative', 'recovered', 
                    'death', 'hospitalized']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Conversion de la date avec gestion des erreurs
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    
    # Renommage des colonnes pour cohérence
    df.rename(columns={
        'state': 'etat',
        'positive': 'cas_positifs',
        'negative': 'cas_negatifs',
        'death': 'deces',
        'hospitalizedCurrently': 'hospitalises_actuels',
        'totalTestResults': 'resultats_tests_totaux'
    }, inplace=True)
    
    return df


# Fonction de prétraitement pour le fichier us_states_covid19_daily.csv
def preprocess_us_states_covid19_daily():
    file_path = "data/covid_csv/us_states_covid19_daily.csv"
    df = pd.read_csv(file_path)
    
    # Conversion de la colonne 'date' en format datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    
    # Conversion des colonnes numériques
    numeric_cols = ['positive', 'negative', 'hospitalizedCurrently', 
                    'hospitalizedCumulative', 'death', 'totalTestResults']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Renommage des colonnes pour plus de clarté
    df.rename(columns={
        'state': 'etat',
        'positive': 'cas_positifs',
        'negative': 'cas_negatifs',
        'death': 'deces',
        'hospitalizedCurrently': 'hospitalises_actuels',
        'totalTestResults': 'resultats_tests_totaux'
    }, inplace=True)
    
    return df


# Fonction de prétraitement pour le fichier us_counties_covid19_daily.csv
def preprocess_us_counties_covid19_daily():
    file_path = "data/covid_csv/us_counties_covid19_daily.csv"
    df = pd.read_csv(file_path)
    
    # Conversion de la colonne 'date' en format datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Conversion des colonnes numériques
    numeric_cols = ['cases', 'deaths']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Renommage des colonnes pour plus de clarté
    df.rename(columns={
        'cases': 'cas',
        'deaths': 'deces'
    }, inplace=True)
    
    return df


# Prétraitement des fichiers CSV
us_covid19_daily_df = preprocess_us_covid19_daily()
us_states_covid19_daily_df = preprocess_us_states_covid19_daily()
us_counties_covid19_daily_df = preprocess_us_counties_covid19_daily()


# Aperçu rapide des données nettoyées (désactivé dans ce cas)
# print(us_covid19_daily_df.head())
# print(us_states_covid19_daily_df.head())
# print(us_counties_covid19_daily_df.head())


### Cellule 4 - Graphique avec facettes et régression linéaire ###
data_to_plot = us_states_covid19_daily_df[
    (us_states_covid19_daily_df['cas_positifs'] > 0) & 
    (us_states_covid19_daily_df['deces'] > 0)
].copy()

data_to_plot.dropna(subset=['cas_positifs', 'deces'], inplace=True)

# Extraction année/mois pour les facettes
data_to_plot['annee'] = data_to_plot['date'].dt.year
data_to_plot['mois'] = data_to_plot['date'].dt.month

fig = px.scatter(
    data_to_plot,
    x="cas_positifs",
    y="deces",
    facet_row="annee",
    facet_col="mois",
    color="etat",
    trendline="ols",
    labels={
        "cas_positifs": "Cas positifs",
        "deces": "Décès",
        "etat": "État"
    },
    title="Relation cas/décès par état (2020-2025)"
)

fig.update_layout(height=1200, width=1500)
fig.show()


### Cellule 7 - Graphique à barres interactif ###
us_covid19_daily_df.sort_values('date', inplace=True)

# Calcul de la hausse quotidienne des cas positifs
us_covid19_daily_df['hausse_quotidienne'] = us_covid19_daily_df['cas_positifs'].diff()

last_30_days = us_covid19_daily_df.dropna().tail(30)

fig_bar = px.bar(
    last_30_days,
    x='date',
    y='hausse_quotidienne',
    color='hausse_quotidienne',
    color_continuous_scale='Viridis',
    labels={'hausse_quotidienne': "Nouveaux cas", "date": "Date"},
    title="Hausse quotidienne des cas positifs"
)

fig_bar.update_layout(
    hovermode='x unified',
    xaxis_tickangle=-45,
)
fig_bar.show()


### Cellule 22 - Carte choroplèthe ###
latest_date = us_states_covid19_daily_df['date'].max()
latest_data = us_states_covid19_daily_df[us_states_covid19_daily_df['date'] == latest_date]

fig_map = px.choropleth(
    latest_data,
    locations='etat',
    locationmode='USA-states',
    color='hospitalises_actuels',
    scope='usa',
    color_continuous_scale='OrRd',
)

fig_map.update_layout(title=f'Hospitalisations actuelles par état ({latest_date.date()})')
fig_map.show()


### Cellules 24-27 - Animation spatio-temporelle ###
filtered_data = us_states_covid19_daily_df[
    us_states_covid19_daily_df['etat'].isin(
        us_states_covid19_daily_df.groupby('etat')['cas_positifs'].max().nlargest(15).index
)].copy()

filtered_data.sort_values('date', inplace=True)
filtered_data["date_str"] = filtered_data["date"].dt.strftime('%Y-%m-%d')

fig_anim = px.scatter(
    filtered_data,
    x="cas_positifs",
    y="deces",
    animation_frame="date_str",
)

fig_anim.show()
