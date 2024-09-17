# cub
#python tools/preprocess/make_classid_classname.py --dataset_name cub --classes_path ../../data/cub/CUB_200_2011/classes.txt
#python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# moe
#python tools/preprocess/make_data_dic_imagenetstyle.py --images_path ../../data/moe/data/ --save_name moe.csv
#python tools/preprocess/data_split.py --df_path ../../data/moe/moe.csv --train_percent 0.8 --save_name_train train_val.csv --save_name_test test.csv
#python tools/preprocess/data_split.py --df_path ../../data/moe/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val
#python tools/preprocess/data_split.py --df_path data/moe/train_val.csv --train_percent 1.0 --save_name_train train_val_5 --save_name_test nothing --max_images_per_class 5

# dafb
#python tools/preprocess/data_split.py --df_path ../../data/daf/train_val.csv --train_percent 0.05 --save_name_train train --save_name_test val

# aircraft
#python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# cars
#python tools/preprocess/data_split.py --df_path ../../data/cars/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# food
#python tools/preprocess/data_split.py --df_path ../../data/food/food-101/train_val.csv --train_percent 0.2 --save_name_train train --save_name_test val

# pets
#python tools/preprocess/data_split.py --df_path ../../data/pets/oxford-iiit-pet/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# dogs
#python tools/preprocess/data_split.py --df_path ../../data/dogs/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# nabirds
#python tools/preprocess/data_split.py --df_path ../../data/nabirds/nabirds/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# flowers doesn't need to split as it has a default val split

# ncfm was already created in its own script

# vegfru
#python tools/preprocess/data_split.py --df_path ../../data/vegfru/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# inat17
#python tools/preprocess/data_split.py --df_path ../../data/inat17/train_val.csv --train_percent 0.2 --save_name_train train --save_name_test val

# cotton
#python tools/preprocess/data_split.py --df_path ../../data/cotton/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# soyageing
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# soygene
#python tools/preprocess/data_split.py --df_path ../../data/soygene/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# soyglobal
#python tools/preprocess/data_split.py --df_path ../../data/soyglobal/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# soylocal
#python tools/preprocess/data_split.py --df_path ../../data/soylocal/train_val.csv --train_percent 0.8 --save_name_train train --save_name_test val

# soyageing r1/r3/r4/r5/r6
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val_R1.csv --train_percent 0.8 --save_name_train train_R1 --save_name_test val_R1
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val_R3.csv --train_percent 0.8 --save_name_train train_R3 --save_name_test val_R3
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val_R4.csv --train_percent 0.8 --save_name_train train_R4 --save_name_test val_R4
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val_R5.csv --train_percent 0.8 --save_name_train train_R5 --save_name_test val_R5
#python tools/preprocess/data_split.py --df_path ../../data/soyageing/train_val_R6.csv --train_percent 0.8 --save_name_train train_R6 --save_name_test val_R6
