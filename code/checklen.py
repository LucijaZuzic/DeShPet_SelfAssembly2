import numpy as np
from utils import DATA_PATH
import matplotlib.pyplot as plt

SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()

suma = 0

yes_num = 0
no_num = 0

all_yes = 0
all_no = 0

all_lens = {}
all_len_list = []

all_yes_lens = {}
all_yes_len_list = []

all_no_lens = {}
all_no_len_list = []

lens = {}
len_list = []

yes_lens = {}
yes_len_list = []

no_lens = {}
no_len_list = []

for pep in SA_data:
    peptide = [pep, SA_data[pep]]
    if peptide[1] != "1" and peptide[1] != "0":
        continue
    if len(peptide[0]) not in all_lens:
        all_lens[len(peptide[0])] = 1
        all_len_list.append(len(peptide[0]))
    else:
        all_lens[len(peptide[0])] += 1
    if peptide[1] == "1":
        all_yes += 1
        if len(peptide[0]) not in all_yes_lens:
            all_yes_lens[len(peptide[0])] = 1
            all_yes_len_list.append(len(peptide[0]))
        else:
            all_yes_lens[len(peptide[0])] += 1
    else:
        all_no += 1
        if len(peptide[0]) not in all_no_lens:
            all_no_lens[len(peptide[0])] = 1
            all_no_len_list.append(len(peptide[0]))
        else:
            all_no_lens[len(peptide[0])] += 1
    if len(peptide[0]) > 24:
        continue
    suma += 1
    if len(peptide[0]) not in lens:
        lens[len(peptide[0])] = 1
        len_list.append(len(peptide[0]))
    else:
        lens[len(peptide[0])] += 1
    if peptide[1] == "1":
        yes_num += 1
        if len(peptide[0]) not in yes_lens:
            yes_lens[len(peptide[0])] = 1
            yes_len_list.append(len(peptide[0]))
        else:
            yes_lens[len(peptide[0])] += 1
    else:
        no_num += 1
        if len(peptide[0]) not in no_lens:
            no_lens[len(peptide[0])] = 1
            no_len_list.append(len(peptide[0]))
        else:
            no_lens[len(peptide[0])] += 1

new_all_lens = {}
for i in sorted(all_lens.keys()):
    new_all_lens[i] = all_lens[i]

new_all_no_lens = {}
for i in new_all_lens.keys():
    if i in all_no_lens:
        new_all_no_lens[i] = all_no_lens[i]
    else:
        new_all_no_lens[i] = 0

plt.rcParams.update({"font.size": 14})

plt.figure(figsize=(30, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=8)
plt.bar(
    range(len(new_all_lens.keys())), new_all_lens.values(), color="#96c1ff", label="SA"
)
plt.bar(
    range(len(new_all_no_lens.keys())),
    new_all_no_lens.values(),
    color="#ff8884",
    label="NSA",
)
plt.xticks(ticks=range(len(new_all_lens.keys())), labels=new_all_lens.keys())
plt.title("Sample peptides by self-assembly category and length")
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of sample peptides\nof a given length")
plt.legend()
plt.savefig(DATA_PATH + "peptide_all.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 22})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
# plt.title("Input peptide sequences")
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number.png", bbox_inches="tight")
plt.show()
plt.close()

sorted_lens_keys = []
sorted_lens_vals = []
sorted_no_lens_keys = []
sorted_no_lens_vals = []
for k in sorted(lens.keys()):
    sorted_lens_keys.append(k)
    sorted_lens_vals.append(lens[k])
for nk in sorted(no_lens.keys()):
    sorted_no_lens_keys.append(nk)
    sorted_no_lens_vals.append(no_lens[nk])

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_line.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_line_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_line.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_line_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_dot.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_dot_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_dot.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_dot_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_dot_line.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
minik = min(min(list(lens.keys())), min(list(no_lens.keys())))
maxik = max(max(list(lens.keys())), max(list(no_lens.keys())))
print(minik, maxik)
maxv_arr = []
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = max(valy, valn)
    maxv_arr.append(maxv)
for ul in range(minik, maxik + 1):
    valy = -1
    valn = -1
    if ul in lens:
        valy = lens[ul]
    if ul in no_lens:
        valn = no_lens[ul]
    maxv = maxv_arr[ul - minik]
    if ul - minik > 0 and ul - minik < len(maxv_arr) - 1:
        prevmax = maxv_arr[ul - minik - 1]
        nextmax = maxv_arr[ul - minik + 1]
        avgmax1 = prevmax + (nextmax - prevmax) * 3 / 4
        avgmax2 = prevmax + (nextmax - prevmax) * 1 / 4
        maxv = max(max(avgmax1, avgmax2), maxv)
    if valy - valn > 0 and not valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valy), color="#2e85ff")
    if not valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
    if valy - valn > 0 and valn > 0:
        plt.text(ul - 0.5, maxv + 23, str(valy - valn), color="#2e85ff")
        plt.text(ul - 0.5, maxv + 5, str(valn), color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_number_dot_line_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.bar(lens.keys(), lens.values(), color="#96c1ff", label="SA")
plt.bar(no_lens.keys(), no_lens.values(), color="#ff8884", label="NSA")
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_dot_line.png", bbox_inches="tight")
plt.show()
plt.close()

plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 5))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter(sorted_lens_keys, sorted_lens_vals, color="#2e85ff", label="SA")
plt.scatter(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a", label="NSA")
plt.plot(sorted_lens_keys, sorted_lens_vals, color="#2e85ff")
plt.plot(sorted_no_lens_keys, sorted_no_lens_vals, color="#ff120a")
plt.xticks(ticks=range(1, max(len_list) + 1), labels=range(1, max(len_list) + 1))
plt.ylim(0, 230)
plt.xlabel("Length of peptide sequence")
plt.ylabel("Number of peptides")
plt.legend()
plt.savefig(DATA_PATH + "peptide_used_dot_line_no_bar.png", bbox_inches="tight")
plt.show()
plt.close()

print(len(SA_data), all_yes, all_no)

output_dict = {}
for i in sorted(all_lens.keys()):
    output_dict[i] = all_lens[i]
print(output_dict)

output_dict = {}
for i in sorted(all_yes_lens.keys()):
    output_dict[i] = all_yes_lens[i]
print(output_dict)

output_dict = {}
for i in sorted(all_no_lens.keys()):
    output_dict[i] = all_no_lens[i]
print(output_dict)

print(suma, yes_num, no_num)

output_dict = {}
for i in sorted(lens.keys()):
    output_dict[i] = lens[i]
print(output_dict)

output_dict = {}
for i in sorted(yes_lens.keys()):
    output_dict[i] = yes_lens[i]
print(output_dict)

output_dict = {}
for i in sorted(no_lens.keys()):
    output_dict[i] = no_lens[i]
print(output_dict)

output_dict = {}
for i in sorted(all_lens.keys()):
    if i not in lens:
        output_dict[i] = all_lens[i]
print(output_dict)
print(list(output_dict.keys()))
