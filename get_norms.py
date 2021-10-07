import argparse
import json
import pickle

import uproot

parser = argparse.ArgumentParser(
    description="Get the MC event yields and apply lumi for each input dataset."
)

parser.add_argument(
    "--fileset",
    type=str,
    help="The fileset extract data from.",
)

parser.add_argument(
    "--luminosity",
    type=float,
    default=200.87,
    help="The integrated luminosity to scale to in picobarns.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    result = {}
    with open(args.fileset) as fin:
        fileset = json.load(fin)
        for dset, info in fileset.items():
            for afile in info["files"]:
                genweights = uproot.open(afile)["hGenWeights"]
                if dset not in result:
                    result[dset] = 0.0
                result[dset] += genweights.to_numpy()[0].sum()

    for dset in result.keys():
        result[dset] = args.luminosity / result[dset]

    outname = ".".join([args.fileset.split(".")[0], "scales.pkl"])
    with open(outname, "wb+") as fout:
        pickle.dump(result, fout)
