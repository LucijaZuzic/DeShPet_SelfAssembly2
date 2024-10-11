import pandas as pd

pd1 = pd.read_csv("my_merged_hex_alt_20.csv", )
pd2 = pd.read_csv("Source_Data_ExtendedDataTable1.csv")
colsmatch = ["Seed", "Test", "Sequence", "Label"]

dict_create = dict()
for ix in range(len(pd2["Seed"])):
    seedcurr = pd2["Seed"][ix]
    testcurr = pd2["Test"][ix]
    seqcurr = pd2["Sequence"][ix]
    labelcurr = pd2["Label"][ix]
    if seedcurr not in dict_create:
        dict_create[seedcurr] = dict()
    if testcurr not in dict_create[seedcurr]:
        dict_create[seedcurr][testcurr] = dict()
    if seqcurr not in dict_create[seedcurr][testcurr]:
        dict_create[seedcurr][testcurr][seqcurr] = dict()
    if labelcurr not in dict_create[seedcurr][testcurr][seqcurr]:
        dict_create[seedcurr][testcurr][seqcurr][labelcurr] = dict()
    for coluse in pd2:
        if coluse not in colsmatch:
            dict_create[seedcurr][testcurr][seqcurr][labelcurr][coluse] = pd2[coluse][ix]

dictnew = dict()
for coluse in pd1:
    dictnew[coluse] = pd1[coluse]
for coluse in pd2:
    if coluse not in colsmatch:
        dictnew[coluse] = []

for ix in range(len(dictnew["Seed"])):
    seedcurr = dictnew["Seed"][ix]
    testcurr = dictnew["Test"][ix]
    seqcurr = dictnew["Sequence"][ix]
    labelcurr = dictnew["Label"][ix]
    for coluse in pd2:
        if coluse not in colsmatch:
            dictnew[coluse].append(dict_create[seedcurr][testcurr][seqcurr][labelcurr][coluse])

df_new_original = pd.DataFrame(dictnew)
df_new_original.to_csv("mine_plus_other.csv", index = False)

writer = pd.ExcelWriter("Source_Data_TableS1.4.xlsx", engine = 'openpyxl', mode = "w")
df_new_original.to_excel(writer, sheet_name = "TableS1.4", index = False)
writer.close()