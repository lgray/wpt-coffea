from typing import Any, Dict

import awkward
import hist
import numpy
import vector
from coffea import processor

vector.register_awkward()


class WpTProcessor(processor.ProcessorABC):  # type: ignore
    def __init__(
        self,
    ) -> None:
        self.recoil_systs = numpy.array([1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        self.sfweight_syst = numpy.array([0, 1, 2, 3, 4, 6, 7])

    def build_4mom(self, events: awkward.Array, name: str) -> awkward.Array:
        if name.find("met") > -1 or name.find("Met") > -1:
            return awkward.zip(
                {"rho": events[name], "phi": events[f"{name}Phi"]},
                with_name="Momentum2D",
            )
        cartesian = awkward.zip(
            {part: events[f"{name}_{part}"] for part in ("px", "py", "pz", "E")},
            with_name="Momentum4D",
        )
        return cartesian.to_rhophietatau()

    def get_histograms(self) -> Dict[Any, Any]:
        lepeta_bins = [0, 1.0, 1.4442, 3.0]  # drop last two bins for muons!
        wpt_bins = [0, 8.0, 16.0, 24.0, 32.0, 40.0, 50.0, 70.0, 100.0]
        mt_min, mt_max, mt_bins = 0.0, 120.0, 12

        return {
            "analysis": (
                hist.Hist.new.StrCategory([], name="dataset", growth=True)
                .StrCategory([], name="systematic", growth=True)
                .IntCategory([-1, 1], name="charge")
                .Regular(mt_bins, mt_min, mt_max, name="mt")
                .Variable(lepeta_bins, name="abseta")
                .Variable(wpt_bins, name="ptW")
                .Variable(wpt_bins, name="ptW_true")
                .Weight()
            ),
        }

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        out = self.get_histograms()

        lep = self.get_4mom(events, "lep")

        wSR_cut = (numpy.abs(lep.eta) < 2.4) & (lep.pt > 25.0)

        # cut down to just the events we're interested in plotting
        events = events[wSR_cut]
        lep = lep[wSR_cut]
        wpt_true = self.get_4mom(events, "genV")

        met = awkward.zip(
            {"rho": events.metVars[:, 1], "phi": events.metVarsPhi[:, 1]},
            with_name="Momentum2D",
        )

        wpt = -1.0 * (lep + met)

        mt = numpy.sqrt(
            2 * lep.pt * events.metVars * (1.0 - numpy.cos(lep.phi - events.metVarsPhi))
        )

        out["analysis"].fill(
            events.metadata.dataset,
            "test",
            events.q,
            mt[:, 0],
            wpt.pt,
            wpt_true.pt,
            weight=1.0,
        )

        return out

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
