cd scraper || { echo "Directory 'scraper' not found. Exiting."; exit 1; }


python3 beau_categories.py || { echo "Failed to execute beau_categories.py. Exiting."; exit 1; }
python3 beau_comp.py || { echo "Failed to execute beau_comp.py. Exiting."; exit 1; }
python3 beau_all_stars.py || { echo "Failed to execute beau_all_stars.py. Exiting."; exit 1; }
python3 parse_reviews.py || { echo "Failed to execute parse_reviews.py. Exiting."; exit 1; }

echo "All scripts executed successfully."