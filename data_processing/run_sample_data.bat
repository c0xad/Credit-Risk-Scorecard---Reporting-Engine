@echo off
echo Generating sample data to CSV...
python generate_sample_data.py --num_customers 20 --export_only --csv_dir ./sample_output
echo Done!
pause 