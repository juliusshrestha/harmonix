import splitfolders

splitfolders.ratio("D:\Projects\music\Music_Seg\harmonix\Dataset", # The location of dataset
                   output="D:\\Projects\\music\\Music_Seg\\harmonix\\\data_train_test\\", # The output location
                   seed=42, # The number of seed
                   ratio=(.7, .2, .1), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )