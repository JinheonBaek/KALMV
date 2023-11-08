python evaluator.py --data_name WebQuestions --data_type KGQA --knowledge_base kg --model_type flan --model_name_or_path google/flan-t5-base --batch_size 32 --verifier_num_epochs 30 --verifier_batch_size 6 --ensemble True --exp_name KALMV_Rectify-0
python evaluator.py --data_name WebQuestions --data_type KGQA --knowledge_base kg --model_type flan --model_name_or_path google/flan-t5-base --batch_size 32 --verifier_num_epochs 30 --verifier_batch_size 6 --ensemble True --edit_output True --num_edits 1 --exp_name KALMV_Rectify-1

python evaluator.py --data_name Mintaka --data_type KGQA --knowledge_base kg --model_type flan --model_name_or_path google/flan-t5-base --batch_size 32 --verifier_num_epochs 10 --verifier_batch_size 6 --ensemble True --exp_name KALMV_Rectify-0
python evaluator.py --data_name Mintaka --data_type KGQA --knowledge_base kg --model_type flan --model_name_or_path google/flan-t5-base --batch_size 32 --verifier_num_epochs 10 --verifier_batch_size 6 --ensemble True --edit_output True --num_edits 1 --exp_name KALMV_Rectify-1