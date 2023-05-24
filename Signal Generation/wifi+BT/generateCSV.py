import numpy as np


list_of_snrs = np.arange(-8,32,2)
list_of_amps = np.logspace(0,-2,3)
#print(list_of_snrs[-1])
#print(list_of_amps[2])
datafolder = 'Signal Generation/wifi_plus_BT/data/'


for amp in list_of_amps:
    for snr in list_of_snrs:
        csv_path = datafolder + str(snr) + 'dB_amp' + str(int(amp*100)) + '.csv'
        csv_path = csv_path.replace('-','minus')
        print(csv_path)
        with open(csv_path, 'x') as creating_new_csv_file: 
            pass

#with open('sample.csv', 'w') as creating_new_csv_file: 
#   pass 
#print("Empty File Created Successfully")