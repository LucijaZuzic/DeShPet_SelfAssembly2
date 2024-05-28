import os
import matplotlib.pyplot as plt
params_for_model = {"AP": 1, "SP": 7, "AP_SP": 1, "TSNE_SP": 5, "TSNE_AP_SP": 8}
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
for model_name in os.listdir("../seeds/seed_369953070"):
    if "similarity" in model_name:
        continue
    print(model_name.replace("_model_data", "").replace("_data", ""))
    num_par = 3
    if "SP" in model_name:
        num_par *= 3
    #for params_num in range(1, num_par):
    params_num = params_for_model[model_name.replace("_model_data", "").replace("_data", "")]
    plt.figure(figsize=(11.7, 8.3), dpi = 110)
    for seed_val_ix in range(len(seed_list)):
        seed_val = seed_list[seed_val_ix]
        print(seed_val)
        for test_num in range(1, 6):
            plt.subplot(len(seed_list), 5, seed_val_ix * 5 + test_num)
            plt.xlim(0, 71)
            plt.ylim(0, 1)
            plt.xticks([])
            plt.yticks([])
            #plt.title("Seed " + str(seed_val) + " Test " + str(test_num))
            dir_loss = "../seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
            for fold_num in range(1, 6):
                train_loss_file = open(dir_loss + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_loss_params_" + str(params_num) + "_fold_" + str(fold_num) + ".txt", "r")
                train_loss_arr = eval(train_loss_file.readlines()[0])
                val_loss_file = open(dir_loss + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_val_loss_params_" + str(params_num) + "_fold_" + str(fold_num) + ".txt", "r")
                val_loss_arr = eval(val_loss_file.readlines()[0])
                plt.plot(train_loss_arr)
                plt.plot(val_loss_arr)
    plt.show()
    plt.close()

