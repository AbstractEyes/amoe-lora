"""Read-only instruments: probes (what the trained model does), vitals
(the campaign's constants and readouts), plots (figures from the ledger).

`plots` is deliberately NOT re-exported here — it is the only module that
touches matplotlib, and importing it eagerly would make `import
aleph_mnist` fail on a base install. Reach it explicitly:

    from aleph_mnist.diagnostics import plots
"""
from . import probes, vitals
from .vitals import BINDING, CV_BAND, GATE_BAND

__all__ = ["probes", "vitals", "BINDING", "GATE_BAND", "CV_BAND"]
