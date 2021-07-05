# Import Libraries
from os import listdir
from os.path import isfile, join
import shutil

# Root Directory
root_dir = "C:/Users/saite/PycharmProjects/py38"
approach = ['BB','RSI']
methods = ['BPS','BHPS','BPHS']
trade_options = [1,2]

# Function to split train & validation
def train_val_split(source,destination,ap,me,val_ratio=0.2):
    # Get the list of all the files in the directory
    imageslist = [f for f in listdir(source) if isfile(join(source, f))]
    val_size = int(val_ratio*(len(imageslist)))
    # Move some files to the destination directory
    for i in range(0,val_size):
        shutil.move(source+str(imageslist[i]), destination)
    train_images = len([f for f in listdir(source) if isfile(join(source, f))])
    val_images = len([f for f in listdir(destination) if isfile(join(destination, f))])
    print("#####################","Approach:",ap,"& Method:",me,"#####################")
    print("Total Images:",len(imageslist),"Training:",train_images,"Validation:",val_images)
    print("##################### Files Moved Successfully#####################")
    imageslist.clear()

# Set the directories
# Loop for all the approaches
# Source - Param 1
source = str()
# Destination - Param 2
destination = str()

for i in approach:
    # BB Approach
    if i == "BB":
        # Loop for all the methods
        for j in methods:
            # Buy & Sell
            if j == "BPS":
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/train/buy/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/val/buy"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/train/sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/val/sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)

            # Buy and [Hold + Sell]
            elif j == "BHPS":
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/train/buy/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/val/buy"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/train/hold_sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/val/hold_sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
            # [Buy + Hold], Sell
            else:
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/train/buy_hold/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/val/buy_hold"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source, destination, i, j, 0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/train/sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/val/sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
    # RSI Approach
    else:
        # Loop for all the methods
        for j in methods:
            # Buy & Sell
            if j == "BPS":
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/train/buy/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/val/buy"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source, destination, i, j, 0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/train/sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/val/sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)

            # Buy and [Hold + Sell]
            elif j == "BHPS":
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/train/buy/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/val/buy"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source, destination, i, j, 0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/train/hold_sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/val/hold_sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
            # [Buy + Hold], Sell
            else:
                for k in trade_options:
                    if k == 1:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/train/buy_hold/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/val/buy_hold"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source, destination, i, j, 0.2)
                    else:
                        source = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/train/sell/"
                        destination = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/val/sell"
                        # source, destination, val_ratio = 0.2, approach, method
                        train_val_split(source,destination,i,j,0.2)
    print("##################### All Files Moved Successfully#####################")