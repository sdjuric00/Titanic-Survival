import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def histograms(data):
    #Histogrami za određena obeležja
    sns.histplot(data['Age'])
    plt.show()

    sns.histplot(data['Pclass'])
    plt.show()

    sns.histplot(data['Sex'])
    plt.show()

    sns.histplot(data['Survived'])
    plt.show()


def bar_plot_survival_by_embarked(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Embarked', hue='Survived', data=data)
    plt.title('Preziveli po lukama ukrcavanja')
    plt.xlabel('Luka ukrcavanja')
    plt.ylabel('Count')
    plt.show()


def survival_by_class(data):
    pclass1 = data[data["Pclass"] == 1]
    pclass2 = data[data["Pclass"] == 2]
    pclass3 = data[data["Pclass"] == 3]

    survived_p1 = pclass1[pclass1["Survived"] == 1].shape[0]
    survived_p2 = pclass2[pclass2["Survived"] == 1].shape[0]
    survived_p3 = pclass3[pclass3["Survived"] == 1].shape[0]

    not_survived_p1 = pclass1.shape[0] - survived_p1
    not_survived_p2 = pclass2.shape[0] - survived_p2
    not_survived_p3 = pclass3.shape[0] - survived_p3

    plt.figure(figsize=(12, 8))
    labels = ["Survived", "Not Survived"]

    plt.subplot(1, 3, 1)
    sizes_p1 = [survived_p1, not_survived_p1]
    plt.pie(sizes_p1, labels=labels, autopct="%1.1f%%")
    plt.title('Prva klasa')

    plt.subplot(1, 3, 2)
    sizes_p2 = [survived_p2, not_survived_p2]
    plt.pie(sizes_p2, labels=labels, autopct="%1.1f%%")
    plt.title('Druga klasa')

    plt.subplot(1, 3, 3)
    sizes_p3 = [survived_p3, not_survived_p3]
    plt.pie(sizes_p3, labels=labels, autopct="%1.1f%%")
    plt.title('Treca klasa')

    plt.show()


def box_plot_survival_by_sex(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Survived', y='Fare', hue='Survived', data=data)
    plt.title('Distribucija cene karte po stepenu preživljavanja')
    plt.xlabel('Preživljavanje')
    plt.ylabel('Cena karte')
    plt.show()


def scatter_survival_by_price(data):
    # Kreiranje scatter plot-a za prikaz odnosa između starosti i cene karte
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Fare', data=data, hue='Survived', palette='coolwarm', alpha=0.7)

    # Dodavanje naslova i oznaka osi
    plt.title('Odnos između Starosti Putnika i Cene Karte')
    plt.xlabel('Starost')
    plt.ylabel('Cena Karte')

    # Prikaz grafa
    plt.show()


def scatter_by_years(data):
    sns.relplot(data['Age'])
    plt.show()


def correlation_matrix(data):
    correlation_matrix = data.corr()

    # Prikaz korelacione matrice
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matrica korelacije')
    plt.show()


def basic_analysis(data):
    print("\nOsnovna analiza\n")

    # Prikaz prvih 5 redova
    print(data.head(5))

    # Osnovne informacije o skupu
    print(data.info())

    #Jedinstvene vrednosti po obeležju
    print(f"\nJedinstvene vrednosti po obeležju: \n{data.nunique()}")

    print(f"\nKabina: {data['Cabin'].unique()}")
    print(f"Cena karte: {data['Fare'].unique()}")
    print(f"Broj roditelja/dece: {data['Parch'].unique()}")
    print(f"Broj braće/sestara ili supružnika: {data['SibSp'].unique()}")

    # Podešavanje float formata za prikaz decimalnih brojeva
    pd.options.display.float_format = '{:.4f}'.format

    # Osnovne statistike skupa
    print(data.describe())

    # Prikaz kolona koje imaju nedostajuće vrednosti
    missing_columns = data.columns[data.isnull().any()]
    print(f"\nKolone sa nedostajucim vrednostima: {missing_columns}")

    # Procenat nedostajućih vrednosti po kolonama
    missing_percentage = data.isnull().mean() * 100
    print(f"\nProcenat nedostajućih vrednosti po kolonama: {missing_percentage}")


def data_visualisation(data):
    print("\nVizualizacija podataka\n")

    histograms(data)
    bar_plot_survival_by_embarked(data)
    survival_by_class(data)
    scatter_survival_by_price(data)
    scatter_by_years(data)
    box_plot_survival_by_sex(data)
    correlation_matrix(data)


def data_cleaning(data):
    #Odbacivanje nepotrebnih kolona
    cleaned_data = data.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin'])

    # Popunjavanje nedostajućih vrednosti median tehnikom
    cleaned_data['Age'].fillna(cleaned_data['Age'].median(), inplace=True)
    cleaned_data['Fare'].fillna(cleaned_data['Fare'].median(), inplace=True)

    return cleaned_data


def survival_by_sex(data):
    # Grupisanje po polu i preživljavanju, zatim brojanje
    survival_by_sex = data.groupby('Sex')['Survived'].mean() * 100

    # Ispis rezultata
    female_survival_rate = survival_by_sex['female']
    male_survival_rate = survival_by_sex['male']

    print(f'Procenat žena koje su preživele: {female_survival_rate:.2f}%')
    print(f'Procenat muškaraca koji su preživeli: {male_survival_rate:.2f}%')


def survival_by_age_group(data):
    # Kreiranje starosnih grupa
    bins = [0, 15, 45, 65, 100]  # Definišemo intervale za starosne grupe
    labels = ['0-15', '16-45', '46-65', '66+']  # Definišemo oznake za svaku grupu
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

    # Kreiranje pivot tabele kako bi se dobio broj preživelih po starosnim grupama
    age_group_survived = data.groupby('AgeGroup')['Survived'].mean() * 100  # Procenat preživelih

    # Vizualizacija pomoću bar plot-a
    plt.figure(figsize=(8, 6))
    sns.barplot(x=age_group_survived.index, y=age_group_survived.values, palette='Blues_d')

    plt.title('Procenat Preživelih po Starosnim Grupama')
    plt.xlabel('Starosne Grupe')
    plt.ylabel('Procenat Preživelih (%)')
    plt.ylim(0, 100)
    plt.show()


if __name__ == '__main__':
    # Učitavanje podataka iz dataseta
    data = pd.read_csv('data/titanic.csv')

    basic_analysis(data)
    data_visualisation(data)
    data = data_cleaning(data)
    survival_by_sex(data)
    survival_by_age_group(data)







