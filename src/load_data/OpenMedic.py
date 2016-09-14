# -*- coding: utf-8 -*-
"""

Les données doivent être chargées depuis datagouv


Source:
https://www.data.gouv.fr/fr/datasets/open-medic-base-complete-sur-les-depenses-de-medicaments-interregimes/

"""

import os
import pandas as pd

path_OpenMedic = 'D:\data\Medicament\OpenMedic'


def read_OpenMedic(year):
    file_path = os.path.join(path_OpenMedic, 'OPEN_MEDIC_' + str(year) + '.csv')
    
    OpenMedic = pd.read_csv(file_path, sep=';', encoding='cp1252')
    OpenMedic = OpenMedic[OpenMedic['CIP13'] != 9999999999999]
    assert all(OpenMedic.groupby(['CIP13'])['ATC5'].nunique() == 1)
    
    incomplete_ATC = ['ATC' + str(k) for k in range(1, 5)]
    for ATC in incomplete_ATC:
        print('verifie que la colonne ' + ATC + " n'apporte rien par " +
            'rapport à ATC5')
        len_ATC = len(OpenMedic[ATC].iloc[0]) 
        assert all(OpenMedic[ATC] == OpenMedic['ATC5'].str[:len_ATC])
        
    OpenMedic.drop(incomplete_ATC, axis=1, inplace=True)
    return OpenMedic






if __name__ == '__main__':
    year = 2015
    OpenMedic = read_OpenMedic(year)
    print('les fichiers contient ', len(OpenMedic)/1e6, ' millions de lignes')
    liste_cip_13 = [str(x) for x in OpenMedic['CIP13'].unique()]
    
    print('en ', str(year), ' on a ', len(liste_cip_13), ' cip différents')
    path_OpenMedic_cip13_list = os.path.join(path_OpenMedic, 'list_cip13.txt')
    
    with open(path_OpenMedic_cip13_list, 'w') as f:
        for item in liste_cip_13:
            f.write("%s\n" % item)