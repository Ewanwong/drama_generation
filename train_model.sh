# parse and split data
python3 data_process/parse_data.py dracor data/parsed_dracor_speech_by_scenes dracor_data data_split
# generate gold label oulines
python3 data_process/outline_extraction_text_rank.py data_split outline_split

# process data for training
python3 data_process/process_training_data.py data_split/train outline_split/train train_data.pkl
python3 data_process/process_training_data.py data_split/dev outline_split/dev dev_data.pkl
python3 data_process/process_training_data.py data_split/test outline_split/test test_data.pkl

# fine-tune model2
python3 gpt2_fine_tuning.py train_data.pkl dev_data.pkl fine_tuned_model2 model2