from task1_1 import task1_1
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
def task1_2():
    df = task1_1()
 
    df = task1_1()

    sbelt_map = {
        1.0: "Seatbelt Worn",
        8.0: "Seatbelt Not Worn"
    }
    df['HELMET_BELT_WORN_LABEL'] = df['HELMET_BELT_WORN'].map(sbelt_map)

    #for barchart
    df_filtered = df[df['HELMET_BELT_WORN_LABEL'].notna()]

    grouped = df_filtered.groupby(['AGE_GROUP', 'HELMET_BELT_WORN_LABEL']).size().unstack()
    print(grouped)
    
    ax = grouped.plot(kind='bar', figsize=(20,6))
    plt.title("Seatbelt Usage by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    for container in ax.containers:
      ax.bar_label(container)
    plt.legend(title="Seatbelt Status")
    plt.tight_layout()
    plt.savefig("task1_2_age.png")
    plt.close()

    #task1_2_2()
    print("without unstack", df_filtered.groupby(['ROAD_USER_TYPE_DESC_Drivers', 'HELMET_BELT_WORN_LABEL']).size())
    grouped_d = df_filtered.groupby(['ROAD_USER_TYPE_DESC_Drivers', 'HELMET_BELT_WORN_LABEL']).size().unstack()
    grouped_p = df_filtered.groupby(['ROAD_USER_TYPE_DESC_Passengers', 'HELMET_BELT_WORN_LABEL']).size().unstack()
    print(grouped_p)
    
    s_pass =  df_filtered.loc[df_filtered['HELMET_BELT_WORN_LABEL'] == "Seatbelt Worn", 'ROAD_USER_TYPE_DESC_Passengers'].sum()
    print(s_pass)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].pie(grouped_d.loc[True], labels=grouped_d.columns, autopct='%1.1f%%', radius = 1.1)
    axes[0].set_title("Seatbelt Use Drivers")

    axes[1].pie(grouped_p.loc[True], labels=grouped_p.columns, autopct='%1.1f%%', radius = 1.1)
    axes[1].set_title("Seatbelt Use Passengers")

    plt.tight_layout()
    plt.savefig("task1_2_driver.png")
    plt.close()

    #task1_2_3()
    #grouped_f = df_filtered.groupby(['SEATING_POSITION', 'HELMET_BELT_WORN_LABEL']).size()
    #print(grouped_f)
    print(df_filtered['SEATING_POSITION'])
    def front_rear(seat):
      if seat in ['LF', 'CF', 'PL']:
        return 'F'
      elif seat in ['RR', 'CR', 'LR', 'OR']:
        return 'R'
      else:
        return None
    df_filtered['SEATING_POSITION'] = df_filtered['SEATING_POSITION'].apply(front_rear)
   
    filtered = df_filtered[df_filtered['SEATING_POSITION'].notna()]
    print(filtered['SEATING_POSITION'])
    grouped = filtered.groupby(['SEATING_POSITION', 'HELMET_BELT_WORN_LABEL']).size().unstack()
    print(grouped)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].pie(grouped.loc['F'], labels=grouped_d.columns, autopct='%1.1f%%', radius = 1.1)
    axes[0].set_title("Seatbelt Use Front")

    axes[1].pie(grouped.loc['R'], labels=grouped_p.columns, autopct='%1.1f%%', radius = 1.1)
    axes[1].set_title("Seatbelt Use Rear")

    plt.savefig('task1_2_seat.png')
    plt.close()

    return ()
