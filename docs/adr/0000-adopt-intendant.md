# ADR-0000 : Adopt intendant

- **Statut** : accepted
- **Date** : 2026-04-30
- **Stacks concernées** : * (transverse)

## Contexte

Ce repo adopte [intendant](https://github.com/dgrauet/intendant) comme framework
de gouvernance — handbook, audit (palier 2), scaffolder (palier 3). Le fichier
`.intendant.toml` à la racine du repo déclare la stack, le mode de conformité
appliqué, et les exemptions justifiées.

## Décision

- Stack détectée à l'adoption : `python`
- Mode initial : `advisory` (les findings sont rapportés mais ne bloquent rien).
- Toutes les ADRs futures du repo numérotées à partir de 0001.

## Conséquences

- L'auditeur (palier 2 d'intendant) pourra rouler sur ce repo et rapporter
  les écarts vs le baseline.
- Les exemptions doivent être listées dans `.intendant.toml` avec une raison.

## Alternatives considérées

- Ne rien adopter (garder les conventions implicites). Rejeté : la dette
  conventionnelle s'accumule en silence.
- Adopter un autre framework : aucun équivalent multi-stack identifié au
  moment de l'adoption.

## Porte de sortie / révision

- Si intendant ne suit plus l'évolution des outils, basculer en `mode = advisory`
  permanent et reprendre les standards à la main.
- Si un baseline `v2` casse trop de règles : geler à `version = "1"` et planifier
  une migration ciblée.
