"""renfc: Research-and-Education-Network traffic forecasting study.

Pipeline stages (run in order, each one is a script in ``scripts/``):

1. ``coverage``      – scan every raw router file, measure per-day data coverage,
                       pick the replacement router and the common healthy window.
2. ``etl``           – one streaming pass per selected router producing the
                       1-minute packet series, flow-size samples and per-day
                       prefix tables used by the later stages.
3. ``sweep``         – forecasting experiments over aggregation window × lookback
                       (lag) × model × seed, evaluated at several lead times.
4. ``distribution``  – heavy-tail / self-similarity statistics per router and a
                       statistics-driven aggregation-window recommendation.
5. ``commonality``   – cross-router traffic commonality (prefix shares and
                       temporal patterns) tied to the backbone topology.
6. ``figures``       – paper figures and LaTeX tables from the stage outputs.

Every stage writes a short Markdown report to ``<out_dir>/reports/`` that
summarises what it did and what it found.
"""

__version__ = "0.1.0"
