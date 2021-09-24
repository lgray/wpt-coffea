from typing import Any, Dict

import awkward
import hist
import numpy
import vector
from coffea import processor

vector.register_awkward()


class WpTQCDProcessor(processor.ProcessorABC):  # type: ignore
    def __init__(
        self,
    ) -> None:
        pass

    def get_4mom(self, events: awkward.Array, name: str) -> awkward.Array:
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

    def get_histograms(self, dataset: str) -> Dict[Any, Any]:
        lepeta_bins = [0, 1.4442, 2.4]
        wpt_bins = [0, 8.0, 16.0, 24.0, 32.0, 40.0, 50.0, 70.0, 100.0]
        mt_min, mt_max, mt_bins = 0.0, 120.0, 12
        isobins = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

        return {
            dataset: (
                hist.Hist.new.IntCategory([-1, 1], name="charge", label="q")
                .Variable(lepeta_bins, name="abseta", label=r"|\eta|")
                .Regular(mt_bins, mt_min, mt_max, name="mt", label=r"M_{T}^{W} (GeV)")
                .Variable(wpt_bins, name="ptW", label=r"p_{T}^{W} (GeV)")
                .Variable(wpt_bins, name="ptW_true", label=r"True p_{T}^{W} (GeV)")
                .Variable(isobins, name="relIso", label=r"lepton Relative Isolation")
                .Weight()
            ),
        }

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset = events.metadata["dataset"]
        out = self.get_histograms(dataset)

        isMC = events.metadata["data_kind"] == "mc"

        lep = self.get_4mom(events, "lep")

        wSR_cut = (numpy.abs(lep.eta) < 2.4) & (lep.pt > 25.0)

        # cut down to just the events we're interested in plotting
        events = events[wSR_cut]
        lep = lep[wSR_cut]
        wpt_true = self.get_4mom(events, "genV")

        # for MC, take corrected recoil,
        # for data, take the raw recoil
        goodMetVars = events.metVars[:, [1] if isMC else [0]]
        goodMetVarsPhi = events.metVarsPhi[:, [1] if isMC else [0]]

        evtWeight = events.evtWeight[:, 0] if isMC else 1.0

        met = awkward.zip(
            {
                "rho": goodMetVars,
                "phi": goodMetVarsPhi,
            },  # index 0 after selection above is central value
            with_name="Momentum2D",
        )

        wpt = -1.0 * (lep.to_rhophi()[:, None] + met)

        mt = numpy.sqrt(
            2
            * lep.pt[:, None]
            * goodMetVars
            * (1.0 - numpy.cos(lep.phi[:, None] - goodMetVarsPhi))
        )

        relIso = events.relIso

        out[dataset].fill(
            charge=events.q,
            abseta=numpy.abs(lep.eta),
            mt=mt[:, 0],
            ptW=wpt[:, 0].pt,
            ptW_true=wpt_true.pt,
            relIso=relIso,
            weight=evtWeight,
        )

        #for i in range(1, self.sfweight_systs.size):  # skip "main" as it == "cent"
        #    out[dataset].fill(
        #        charge=events.q,
        #        abseta=numpy.abs(lep.eta),
        #        mt=mt[:, 0],  # idx 0 is central value for mt
        #        ptW=wpt[:, 0].pt,
        #        ptW_true=wpt_true.pt,
        #        relIso=relIso,
        #        weight=events.evtWeight[:, self.sfweight_systs[i]],
        #    )

        return out

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
