import os
import matplotlib.pyplot as plt
hyperparameter_numcells = [32, 48, 64]
hyperparameter_kernel_size = [4, 6, 8]
long_list = []
short_list = []
for nc in hyperparameter_numcells:
    short_list.append([2 * nc])
    for ks in hyperparameter_kernel_size:
        long_list.append([nc, ks])
params_for_model = {"AP": 1, "SP": 7, "AP_SP": 1, "TSNE_SP": 5, "TSNE_AP_SP": 8}
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}
for model_name in os.listdir("seeds/seed_305475974"):
    if "similarity" in model_name:
        continue
    #print(model_name.replace("_model_data", "").replace("_data", ""))
    num_par = 3
    if "SP" in model_name:
        num_par *= 3
    for params_num in range(1, num_par + 1):
        #params_num = params_for_model[model_name.replace("_model_data", "").replace("_data", "")]
        plt.rcParams["svg.fonttype"] = "none"
        from matplotlib import rc
        rc('font',**{'family':'Arial'})
        cm = 1/2.54  # centimeters in inches
        #plt.rcParams.update({"font.size": 7})
        SMALL_SIZE = 7
        MEDIUM_SIZE = 7
        BIGGER_SIZE = 7

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.figure(figsize=(22*cm, 26.8*cm), dpi = 300)
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            #print(seed_val)
            for test_num in range(1, 6):
                plt.subplot(len(seed_list), 5, seed_val_ix * 5 + test_num)
                plt.xlim(0, 70)
                plt.ylim(0, 1)
                plt.xticks([])
                plt.yticks([])
                color_list_train = ["red", "green", "blue", "orange", "purple"]
                color_list_val = ["yellow", "cyan", "magenta", "pink", "brown"]
                if seed_val_ix == 0:
                    if test_num == 3:
                        long_title = model_order[model_name.replace("_model_data", "").replace("_data", "")] + " model"
                        if "AP" in model_name and "SP" not in model_name:
                            long_title += " (dense " + str(short_list[params_num - 1][0])
                        if "AP" in model_name and "SP" in model_name:
                            long_title += " (num cells " + str(long_list[params_num - 1][0])
                            long_title += ", kernel size " + str(long_list[params_num - 1][1])
                            long_title += ", dense " + str(long_list[params_num - 1][0] * 2)
                        if "AP" not in model_name and "SP" in model_name:
                            long_title += " (num cells " + str(long_list[params_num - 1][0])
                            long_title += ", kernel size " + str(long_list[params_num - 1][1])
                        plt.title(long_title + ")\nTest " + str(test_num))
                    else:
                        plt.title("Test " + str(test_num))
                if test_num == 1:
                    plt.ylabel("Seed " + str(seed_val))
                dir_loss = "seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
                for fold_num in range(1, 6):
                    train_loss_file = open(dir_loss + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_loss_params_" + str(params_num) + "_fold_" + str(fold_num) + ".txt", "r")
                    train_loss_arr = eval(train_loss_file.readlines()[0])
                    val_loss_file = open(dir_loss + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_val_loss_params_" + str(params_num) + "_fold_" + str(fold_num) + ".txt", "r")
                    val_loss_arr = eval(val_loss_file.readlines()[0])
                    plt.plot(train_loss_arr, c = color_list_train[fold_num - 1], label = "Training fold " + str(fold_num))
                    plt.plot(val_loss_arr, c = color_list_val[fold_num - 1], label = "Validation fold " + str(fold_num))
                if test_num == 1 and seed_val_ix == len(seed_list) - 1:
                    plt.legend(loc = "lower left", ncol = 5, bbox_to_anchor = (0, -0.6*cm))
        file_name = "review/" + model_name.replace("_model_data", "").replace("_data", "") + "_params_" + str(params_num)
        if not os.path.isdir("review/"):
            os.makedirs("review/")
        print(file_name)
        plt.subplots_adjust(wspace = 0, hspace = 0)
        plt.savefig(file_name, bbox_inches = "tight")
        plt.savefig(file_name + ".svg", bbox_inches = "tight")
        plt.savefig(file_name + ".pdf", bbox_inches = "tight")
        plt.close()