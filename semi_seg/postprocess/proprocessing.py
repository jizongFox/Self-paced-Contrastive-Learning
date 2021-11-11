# adding postprocessing module to produce images.
import matplotlib as mpl
import matplotlib.pyplot as plt

labeled_ratios = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.2, 1.0]

mts = [0.789688726, 0.808321397, 0.828677118, 0.841974099, 0.8583318, 0.867948234, 0.87966605, 0.887115121,
       0.8950]
uda_iic = [0.658286552, 0.80021592, 0.826074898, 0.857801437, 0.867771029, 0.873659591, 0.884436687,
           0.888066709, 0.8950]

ps = [0.345107834, 0.503013064, 0.563559552, 0.720328708, 0.822766721, 0.845397453, 0.864634971,
      0.880039314, 0.8950]

iicmeanteacher_labeled_ratios = [0.02, 0.03, 0.04, 0.05, 0.06,]# 0.07, 0.08, 0.1, 0.2, 1.0]
iicmeanteacher = [0.807905882, 0.820376019, 0.830185831, 0.845082221, 0.860671893, ]#0.869388, 0.874142619, 0.879262706,
                  #0.888023555, 0.8950]

linewith = 1.5

plt.figure(figsize=(5.2, 3.5))
plt.hlines(0.8950, -1, 2, linestyles="dashdot", colors="red", label="Full Supervision")
plt.plot(labeled_ratios, ps, label="Partial Supervision", marker="x", markersize=8, linewidth=linewith)
plt.plot(labeled_ratios, mts, label="Mean Teacher", marker=".", markersize=8, linewidth=linewith)

plt.vlines(0.05, -1, 1, linestyles=":")
plt.plot(labeled_ratios, uda_iic, label="Ours", marker="*", markersize=8, linewidth=linewith)
plt.plot(iicmeanteacher_labeled_ratios, iicmeanteacher, marker="^", markersize=6.5, label="Ours (Mean Teacher)",
         color="lightgreen", linewidth=linewith)
plt.xscale("log")
plt.xticks([0.02, 0.03, 0.05, 0.07, 0.1, 0.2], rotation=0)
plt.gca().get_xaxis().set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.xlim([0.019, 0.5])
plt.ylim([0.7, 0.93])
plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.legend(loc="lower right")
# plt.title("3D mean dice for ACDC dataset with different labeled data ratio")
plt.grid(which="both")
plt.xlabel("Labeled Ratio")
plt.ylabel("3D mean DSC on Validation Set")
# plt.show()
plt.savefig("different_label_ratio.pdf", bbox_inches="tight", format="pdf", dpi=500)
plt.show()
