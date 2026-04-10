# Courbe de Phillips – France (2005–2024)

Petit Projet simple et interactif.
Visualisation interactive de la courbe de Phillips pour la France entre 2005 et 2024, avec une régression linéaire OLS entre le taux de chômage et l’inflation annuelle.

## Objectif

Ce projet présente une illustration simple de la relation empirique entre chômage et inflation.  
Le graphique montre les observations annuelles, ainsi qu’une droite de régression estimée par moindres carrés ordinaires.

## Contenu

- Un graphique interactif Plotly.
- Un nuage de points avec annotations par année.
- Une droite de régression OLS.
- Des indicateurs statistiques résumés.
- Un tableau des données utilisées.
- Une mise en page responsive et lisible sur desktop comme mobile.

## Données

Le projet utilise des données annuelles France 2005–2024 :
- taux de chômage harmonisé en pourcentage,
- inflation annuelle en pourcentage.

Les séries sont présentées comme des valeurs approximatives cohérentes avec des sources OCDE, BCE et INSEE.

## Méthode

Le modèle estimé est :

```text
π_t = α + β · u_t + ε_t
```

avec :
- \( \pi_t \) = inflation annuelle,
- \( u_t \) = taux de chômage,
- \( \alpha \) = constante,
- \( \beta \) = pente estimée,
- \( \varepsilon_t \) = terme d’erreur.

Une régression linéaire OLS est calculée directement en JavaScript afin d’obtenir :
- l’équation estimée,
- le coefficient de détermination \( R^2 \),
- l’écart-type des résidus.

## Fonctionnalités techniques

- HTML/CSS/JavaScript pur.
- Librairie Plotly chargée via CDN.
- Affichage responsive.
- Mise en page centrée et propre.
- Tableau récapitulatif intégré.

## Utilisation

Il suffit d’ouvrir le fichier HTML dans un navigateur moderne.  
Aucune installation ou compilation n’est nécessaire.

## Structure

```text
project/
└── index.html
```

## Technologies utilisées

- HTML5
- CSS3
- JavaScript
- Plotly.js

## Remarque

Ce projet est une visualisation pédagogique et académique.  
Il sert à illustrer une relation macroéconomique classique et ne constitue pas une preuve causale automatique entre chômage et inflation.

## Auteur

Projet personnel d’économétrie appliquée.
