Ensure that the CSV file is located outside the task5 folder, in the same directory that contains the task5 folder.

project_root/
├── A2/
│   └── main.py
│   └── mlp_model.py
│   └── lightgbm_model.py
└── accident.csv
└── person.csv
└── filtered_vehicle.csv



run everything in proprocess.ipynb to generate the required dataset for both models

for lightGBM, run the command: python main.py --model lightgbm
for MLP, run the command: python main.py --model mlp

library used:
scikit-learn==1.6.1
matplotlib==3.10.1
pandas==2.2.3
numpy==2.2.4
seaborn==0.13.2
lightgbm==4.6.0
folium==0.12.0 