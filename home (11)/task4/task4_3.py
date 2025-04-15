import pandas as pd
import json

def task4_3():
    # read vehicle data
    df = pd.read_csv('../vehicle.csv')

    # drop rows with NA make or body style
    df = df.dropna(subset=['VEHICLE_MAKE', 'VEHICLE_BODY_STYLE'])

    # group by make, count unique body styles
    diversity_counts = df.groupby('VEHICLE_MAKE')['VEHICLE_BODY_STYLE'].nunique()

    # top 5 manufacturers
    top5 = diversity_counts.sort_values(ascending=False).head(5)

    # diversity output
    diversity_output = {
        "Manufacturers": top5.index.tolist()
    }
    for make in top5.index:
        diversity_output[make] = int(top5[make])  
        
    with open('task4_3_diversity.json', 'w') as f:
        json.dump(diversity_output, f, indent=4)

    # probability output
    prob_output = {}
    for make in top5.index:
        make_df = df[df['VEHICLE_MAKE'] == make]
        total = len(make_df)
        type_counts = make_df['VEHICLE_BODY_STYLE'].value_counts().head(3)

        prob_output[make] = [
            {
                "Vehicle Type": vtype,
                "Probability": round(count / total, 2)
            }
            for vtype, count in type_counts.items()
        ]

    with open('task4_3_probabilities.json', 'w') as f:
        json.dump(prob_output, f, indent=4)
