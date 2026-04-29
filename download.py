import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def get_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df.to_csv("raw_sleep_data.csv", index=False)
    return df

def clear_data(path_to_csv):
    df = pd.read_csv(path_to_csv)

    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])
    
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop(columns=['Blood Pressure'])
    
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    cat_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    df.to_csv('df_clear.csv', index=False)
    print("Saved to df_clear.csv")
    return True

if __name__ == "__main__":
    get_data()
    clear_data("raw_sleep_data.csv")
