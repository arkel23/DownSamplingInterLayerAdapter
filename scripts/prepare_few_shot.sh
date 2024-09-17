python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --max_images_per_class 1 --save_name_train train_val_1 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --max_images_per_class 2 --save_name_train train_val_2 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --max_images_per_class 4 --save_name_train train_val_4 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --max_images_per_class 8 --save_name_train train_val_8 --save_name_test none --train_percent 1
rm ../../data/aircraft/fgvc-aircraft-2013b/data/none.csv

python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --max_images_per_class 1 --save_name_train train_val_1 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --max_images_per_class 2 --save_name_train train_val_2 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --max_images_per_class 4 --save_name_train train_val_4 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --max_images_per_class 8 --save_name_train train_val_8 --save_name_test none --train_percent 1
rm ../../data/cub/CUB_200_2011/none.csv

python tools/preprocess/data_split.py --df_path ../../data/soygene/train_val.csv --max_images_per_class 1 --save_name_train train_val_1 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/soygene/train_val.csv --max_images_per_class 2 --save_name_train train_val_2 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/soygene/train_val.csv --max_images_per_class 4 --save_name_train train_val_4 --save_name_test none --train_percent 1
python tools/preprocess/data_split.py --df_path ../../data/soygene/train_val.csv --max_images_per_class 8 --save_name_train train_val_8 --save_name_test none --train_percent 1
rm ../../data/soygene/none.csv
