import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
import pickle
from coffea.util import load
#from coffea.hist import plot
import re
from collections import OrderedDict
import scipy.optimize
import boost_histogram as bh
import uproot
from hist import Hist, Stack
import colorsys

# load MC weights
lumi_weights = None
with open("filefetcher/aram_samples_lpc.scales.pkl", "rb") as fin:
    lumi_weights = pickle.load(fin)
    lumi_weights["data_mu"] = 1.0
    lumi_weights["data_el"] = 1.0

# load histograms
output = load("hists_wptqcd_filefetcher_aram_samples_lpc_aI.coffea")
#
#for name, h in output.items():
#    print(name, h)

# sale the MC
data_mu = {name: h * lumi_weights[name] for name, h in output.items() if name.find("data") > -1 and name[-2:] == "mu"}
mc_mu = {name: h * lumi_weights[name] for name, h in output.items() if name.find("data") == -1 and name[-2:] == "mu"}

# group the histograms in different channels
mc_mu_combined = OrderedDict()
mc_mu_combined['VV_mu'] = mc_mu['wz_mu'] + mc_mu['ww_mu'] + mc_mu['zz_mu']
mc_mu_combined['ttbar_mu'] = mc_mu['ttbar_dilepton_mu'] + mc_mu['ttbar_onelepton_mu'] + mc_mu['ttbar_hadronic_mu']
mc_mu_combined['DY_mu'] = mc_mu['zll_mu']
mc_mu_combined['wtau_mu'] = mc_mu['wtau2_mu'] + mc_mu['wtau1_mu'] + mc_mu['wtau0_mu']
mc_mu_combined['wl_mu'] = mc_mu['wl2_mu'] + mc_mu['wl1_mu'] + mc_mu['wl0_mu']

mts_mu = [h.project("charge", "mt","relIso", "ptW") for _, h in mc_mu_combined.items()]
mts_mu_data = sum(h for h in data_mu.values()).project("charge","mt", "relIso", "ptW")
mts_mu_scaled = Stack(*mts_mu)

nIsoBins = len(data_mu['data_mu'].axes['relIso'])
nMTBins = len(list(mts_mu_data.axes['mt']))
nPTBins = len(list(mts_mu_data.axes['ptW']))
mtBins = list(mts_mu_data.axes['mt'])
isoBins = list(data_mu['data_mu'].axes['relIso'])
ptBins = list(mts_mu_data.axes['ptW'])


# plot the data-MC comparison in the anti-isolated region
# to check the signal contamination
doPlotCR = True
if doPlotCR:
    legends = OrderedDict()
    legends['VV_mu'] = 'Dibosons'
    legends['ttbar_mu'] = r't$\bar{t}$'
    legends['DY_mu'] = 'DrellYan'
    legends['wtau_mu'] = r'W($\tau\nu$)'
    legends['wl_mu'] = r'W($\ell\nu$)'
    for iwpt in range(nPTBins):
        for iso in range(nIsoBins):
            f, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20,10))
            for i, q in enumerate([-1, 1]):
                hep.histplot(mts_mu_data[i,:,iso, iwpt], histtype="errorbar", label=["Data"], color="black", ax=axs[i])
                hep.histplot([x[i,:,iso, iwpt] for x in mts_mu_scaled], stack=True, label=[legends[name] for name, _ in mc_mu_combined.items()], ax=axs[i], histtype="fill")
                handles, labels = axs[i].get_legend_handles_labels()
                handles.insert(0, handles.pop())
                labels.insert(0, labels.pop())
                axs[i].legend(handles, labels, fontsize=15, bbox_to_anchor=(0.95,0.97))
                #hep.cms.label(loc=0, data=True, llabel="Preliminary", ax=axs[iso,i])
                channel = r'$W^{+}\to\mu^{+}\nu$' if q>0 else r'$W^{-}\to\mu^{-}\nu$'
                isobin = '{:.2f} < I < {:.2f}'.format(isoBins[iso][0],isoBins[iso][1])
                iptbin = r'{:.2f} < $p^W_T$ < {:.2f}'.format(ptBins[iwpt][0], ptBins[iwpt][1])
                axs[i].text(0.72, 0.58, channel, fontsize=15, transform=axs[i].transAxes)
                axs[i].text(0.72, 0.53, isobin, fontsize=15, transform=axs[i].transAxes)
                axs[i].text(0.72, 0.48, iptbin, fontsize=15, transform=axs[i].transAxes)
                axs[i].set_xlabel(r'$m_{T}$ [GeV]')
            plt.savefig(f'plots/qcd_antiIsolated_wpt{iwpt}_Iso{iso}_.png', bbox_inches='tight')  
            plt.close()
            #plt.show()


# subtract the mc contribution from data in the CR
mts_mu_mc = sum(list(mts_mu_scaled))
hsubtracted = mts_mu_data + mts_mu_mc * (-1)
# scale the MC normalization by 30% to evaluate the singal
# contamination systematics on the QCD template
hsubtracted_scaled = mts_mu_data + mts_mu_mc * (-1.3)

# normalize the histograms in different isolation bins to the same, so that it would only be the shape effects
# maybe there are more efficient ways to do this instead of looping over all bins
for iwpt in range(nPTBins):
    for i in range(2):
        h = hsubtracted.project("charge", "relIso", "ptW")[i,0,iwpt]
        for iso in range(1, nIsoBins):
            norm = float(h.value / (hsubtracted.project("charge","relIso", "ptW")[i,iso, iwpt].value))
            norm_scaled = float(h.value / (hsubtracted_scaled.project("charge","relIso", "ptW")[i,iso, iwpt].value))

            # hsubtracted[i,:,iso,iwpt] *= norm seems not working
            for imt in range(nMTBins):
                hsubtracted[i,imt,iso,iwpt] *= norm
                hsubtracted_scaled[i,imt,iso,iwpt] *= norm_scaled


# run the linear fit and save th1 to root file
root_file = uproot.recreate("output_qcd_templates.root")

# hard coded iso centers
isocenters = np.array([0.225, 0.275, 0.320, 0.375, 0.425, 0.475, 0.525, 0.575])
isoSR = 0.025

def pol1(x, a, b):
    return (x - isoSR)*a + b

def pol2(x, a, b, c):
    return (x - isoSR)**2 * a + (x-isoSR)*b + c

# run the pol1 fit in different mT bins
# to extrapolate the QCD template
for iwpt in range(nPTBins):
    for i, q in enumerate([-1, 1]):
        print(f"fit for wpt bin {iwpt}, charge bin {i}")
        # to save the extrapolated histogram
        h_extrapolated = bh.Histogram(mts_mu_data.axes['mt'], storage=bh.storage.Weight())
        # to save the extrapolated histogram with MC xsec variations (systematics)
        h_extrapolated_scaled = bh.Histogram(mts_mu_data.axes['mt'], storage=bh.storage.Weight())
        h_extrapolated_scaled_dn = bh.Histogram(mts_mu_data.axes['mt'], storage=bh.storage.Weight())
        
        for imt in range(nMTBins):
            #if imt!=0:
            #    continue
            f, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10,10))
    
            values = np.zeros(nIsoBins-1)
            errors = np.zeros(nIsoBins-1)

            values_scaled = np.zeros(nIsoBins-1)
            errors_scaled = np.zeros(nIsoBins-1)
            
            for iso in range(nIsoBins-1):
                #print(i, imt, iso, hsubtracted[i, imt, iso].value)
                values[iso] = hsubtracted[i, imt, iso, iwpt].value
                # avoid ill-defined 0 value and 0 error
                errors[iso] = max(np.sqrt(hsubtracted[i, imt, iso, iwpt].variance), 1e-6)

                values_scaled[iso] = hsubtracted_scaled[i, imt, iso, iwpt].value
                errors_scaled[iso] = max(np.sqrt(hsubtracted_scaled[i, imt, iso, iwpt].variance), 1e-6)
           
            # fit from the 2nd isolation bin
            # the 1st has too large signal contamination
            popt, pcov = scipy.optimize.curve_fit(pol1, isocenters[1:], values[1:], sigma=errors[1:])
            #print(popt)
            #print(np.sqrt(np.diag(pcov)))
            popt_scaled, pcov_scaled = scipy.optimize.curve_fit(pol1, isocenters[1:], values_scaled[1:], sigma=errors_scaled[1:])
            
            axs.errorbar(isocenters, values, yerr=errors, marker='o', color='black', ls='none',label='Data')
            axs.plot(isocenters[1:], pol1(isocenters[1:], *popt), 'r-', label='Pol1 Fit')

            iso_extrapolated = np.arange(0., 0.25, 0.02)
            axs.plot(iso_extrapolated, pol1(iso_extrapolated, *popt), 'r--',)
            axs.errorbar(isoSR, popt[1], yerr=np.sqrt(np.diag(pcov))[1], marker='o', color='blue', ls='none',markersize=10, label='Extrapolated')
            #hep.cms.label(loc=0, data=True, llabel="Preliminary", ax=axs[ix,i])
            channel = r'$W^{+}\to\mu^{+}\nu$' if q>0 else r'$W^{-}\to\mu^{-}\nu$'
            channelbin = r'{:.0f} < $m_T$ < {:.0f}'.format(mtBins[imt][0],mtBins[imt][1])
            iptbin = r'{:.2f} < $p^W_T$ < {:.2f}'.format(ptBins[iwpt][0], ptBins[iwpt][1])
            axs.text(0.72, 0.58, channel, fontsize=15, transform=axs.transAxes)
            axs.text(0.72, 0.54, channelbin, fontsize=15, transform=axs.transAxes)
            axs.text(0.72, 0.50, iptbin, fontsize=15, transform=axs.transAxes)
            axs.set_xlabel('Lepton Isolation')
            axs.set_ylabel('Bin Contents')
            axs.legend(fontsize=15, bbox_to_anchor=(0.95,0.77))
            axs.set_xlim(0,0.6)
         
            chg = "plus" if q>0 else "minus"
            plt.savefig(f'plots/qcd_extrapolated_wpt{iwpt}_mT{imt}_W{chg}munu.png', bbox_inches='tight')  
            plt.close()
            #plt.show()
            
            
            val_extrapolated = max(popt[1], 0.)
            err_extrapolated = np.sqrt(np.diag(pcov))[1]
            h_extrapolated[imt] = [val_extrapolated, err_extrapolated**2]

            # set the stat uncertainty of the varied QCD shape as 0
            val_extrapolated_scaled = max(popt_scaled[1], 0.)
            h_extrapolated_scaled[imt] = [val_extrapolated_scaled, 0.]
            h_extrapolated_scaled_dn[imt] = [2 * val_extrapolated - val_extrapolated_scaled, 0.]
       
        # save the template and its systematic variation to the root file
        hname = f'qcd_extrapolated_wpt{iwpt}_W{chg}munu'
        root_file[hname] = h_extrapolated
        root_file[hname+"_ScaledMCshapeUp"] = h_extrapolated_scaled
        root_file[hname+"_ScaledMCshapeDown"] = h_extrapolated_scaled_dn
root_file.close()


compareShape = True
if compareShape:
    # plot the extrapolated shape and the original shape in CRs
    def scale_lightness(rgb, scale_l):
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l/(nIsoBins+1)), s = s)
    
    color = matplotlib.colors.ColorConverter.to_rgb("red")
    rgbs = [scale_lightness(color, scale) for scale in range(nIsoBins)]
    
    root_file = uproot.open("output_qcd_templates.root")
    root_file.keys()
    # compare the extrapolated histogram with the ones in the anti-isolated regions
    #
    # might be good to add ratios here
    #
    for iwpt in range(nPTBins):
        for i, q in enumerate([-1, 1]):
            f, axs = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10,10))
            for iso in range(nIsoBins-1):
                isobin = '{:.2f} < I < {:.2f}'.format(isoBins[iso][0],isoBins[iso][1])
                hep.histplot(hsubtracted[i, :, iso, iwpt], histtype="errorbar", label=isobin, color=rgbs[iso])
            
            chg = "plus" if q>0 else "minus"
            hname = f'qcd_extrapolated_wpt{iwpt}_W{chg}munu'
            hep.histplot(root_file[hname], histtype="errorbar", label='Extracted', color='blue')
    
            hep.histplot(root_file[hname+"_ScaledMCshapeUp"], histtype="errorbar", label='Extracted MCUp', color='green')
               
            axs.legend(fontsize=15, bbox_to_anchor=(0.95,0.97))
            
            axs.set_xlabel(r'$m_{T}$ [GeV]')
            axs.set_ylabel('A.U.')
            axs.legend(fontsize=15, bbox_to_anchor=(0.95,0.77))
            
            channel = r'$W^{+}\to\mu^{+}\nu$' if q>0 else r'$W^{-}\to\mu^{-}\nu$'
            iptbin = r'{:.0f} < $p^W_T$ < {:.0f}'.format(ptBins[iwpt][0], ptBins[iwpt][1])
            axs.text(0.72, 0.88, channel, fontsize=15, transform=axs.transAxes)
            axs.text(0.72, 0.84, iptbin, fontsize=15, transform=axs.transAxes)
            chg = "plus" if q>0 else "minus"
            plt.savefig(f'plots/qcd_compShape_wpt{iwpt}_W{chg}munu.png', bbox_inches='tight')  
            plt.close()
            #plt.show()
    root_file.close()
