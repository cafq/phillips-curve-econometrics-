import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- 1) Charger datasets OCDE sans API ---
url_infl = "https://raw.githubusercontent.com/hedonometer/cpi-data/master/FRANCE_CPI.csv"
url_unemp = "https://raw.githubusercontent.com/datasets/unemployment/master/data/FR.csv"

infl = pd.read_csv(url_infl)
unemp = pd.read_csv(url_unemp)

infl = infl.rename(columns={"date": "date", "value": "inflation"})
unemp = unemp.rename(columns={"Date": "date", "Value": "unemployment"})

# Harmoniser format des dates
infl["date"] = infl["date"].astype(str).str[:7]
unemp["date"] = unemp["date"].astype(str).str[:7]

# Merge
df = infl.merge(unemp, on="date")
df = df.dropna()

# Convertir en float
df["inflation"] = df["inflation"].astype(float)
df["unemployment"] = df["unemployment"].astype(float)

# --- 2) Régression OLS ---
X = sm.add_constant(df["unemployment"])
y = df["inflation"]
model = sm.OLS(y, X).fit()

print(model.summary())

# --- 3) Graphique ---
plt.figure(figsize=(8,6))
plt.scatter(df["unemployment"], df["inflation"], alpha=0.7, label="Données OCDE")
plt.xlabel("Chômage (%)")
plt.ylabel("Inflation (%)")
plt.title("Courbe de Phillips — France")

# Droite de régression
x_vals = df["unemployment"]
y_pred = model.params["const"] + model.params["unemployment"] * x_vals
plt.plot(x_vals, y_pred, color="red", label="Régression OLS")

plt.legend()
plt.tight_layout()

plt.savefig("phillips_curve_france.png")
plt.close()
