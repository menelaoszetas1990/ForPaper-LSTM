import pickle

dbfile = open('./2-Plots_for_Paper/Figure1_Training_Loss_for_Learning_Rates_per_Epochs_per_Dataset/pickle/Figure_1_ship_1', 'rb')
db = pickle.load(dbfile)
# for record in db:
print(db)
dbfile.close()

# dbfile = open('./2-Plots_for_Paper/Figure2_Training_Loss_for_Dense_Layers_per_Epochs_per_Dataset/pickle/Figure_2_ship_0', 'rb')
# db = pickle.load(dbfile)
# for record in db:
#     print(record)
# dbfile.close()
