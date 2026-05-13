# PARTEA Server

Backend server for **PARTEA** (Privacy-Aware Time-to-Event Analysis), a federated platform for cross-institutional survival analysis without sharing raw patient data.

## What this is

PARTEA enables multiple hospitals or research institutes to jointly run survival analyses (Kaplan-Meier, Cox regression, federated survival SVMs) on their combined patient cohorts, while keeping all individual-level data on-premise.

This repository contains the **coordination server**, which:
- Orchestrates federated computation rounds across participating clients
- Aggregates partial statistics (never raw patient data)

For the user-facing web interface, see [federated-partea/webapp](https://github.com/federated-partea/webapp).

## Tech Stack

- Python, Django
- PostgreSQL
- Docker
- Lifelines

## Reference

Späth et al. (2022). *Privacy-aware multi-institutional time-to-event studies*. PLOS Digital Health.
[Link to paper](https://doi.org/10.1371/journal.pdig.0000101)

## License

See [LICENSE](LICENSE).
