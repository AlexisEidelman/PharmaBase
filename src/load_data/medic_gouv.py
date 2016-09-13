# -*- coding:cp1252 -*-

'''
Télécharge les données depuis la base publique des médicaments :
https://www.data.gouv.fr/fr/datasets/base-de-donnees-publique-des-medicaments-base-officielle/

'''
import pandas as pd
import numpy as np
import re
import os
import datetime as dt

pd.set_option('max_colwidth', 100)

path_gouv = "D:\data\Medicament\medicament_gouv"
maj_bdm = 'maj_20141128' # 'maj_20140915122241' # maj_20141128


dico_variables = dict(
    bdpm=['CIS', 'Nom', 'Forme', 'Voies', 'Statut_AMM', 'Type_AMM', 'Etat',
          'Date_AMM', 'Statut_BDM', 'Num_Europe', 'Titulaires', 'Surveillance'],
    CIP_bdpm=['CIS', 'CIP7', 'Label_presta', 'Statu_admin_presta',
              'etat_commercialisation', 'Date_declar_commerc', 'CIP',
              'aggrement_collectivites', 'Taux_rembours', 'Prix',
              'indic_droit_rembours'],
    CPD_bdpm=['CIS', 'Prescription'],
    GENER_bdpm=['Id_Groupe', 'Nom_Groupe', 'CIS', 'Type', 'Num_Tri'],
    COMPO_bdpm=['CIS', 'Element_Pharma', 'Code_Substance', 'Nom_Substance',
                'Dosage', 'Ref_Dosage', 'Nature_Composant',
                'Substance_Fraction'],
    HAS_SMR_bdpm=['CIS', 'HAS', 'Evalu', 'Date_SMR', 'Valeur_SMR', 'Libelle_SMR'],
    HAS_ASMR_bdpm=['CIS', 'HAS', 'Evalu', 'Date_ASMR', 'Valeur_ASMR', 'Libelle_ASMR']
    )

unite_standard = ['ml', 'mg', 'litre']
element_standard = [u'comprimé', u'gélule', u'capsule', u'flacon', u'ampoule',
                    u'dispositif', u'lyophilisat', u'pastille', u'seringue',
                    u'sachet-dose', u'suppositoire', u'dose', u'ovule',
                    u'sachet', u'gomme', u'tube', u'bâton', u'creuset', u'insert',
                    u'récipient', u'poche', u'cartouche', u'pression', u'film',
                    u'cm^2', u'générateur', u'stylo', u'emplâtre',
                    u'goutte', u'anneau', u'éponge', u'pâte', u'compresse',
                    u'implant', u'récipient', u'pot', u'bouteille', u'unité',
                    u'pilule', u'seringue préremplie']
                    # u'mole',  pour les gaz
            #contenants = ['plaquette','flacon','tube', 'récipient', 'sachet',
#              'cartouche', 'boite', 'pochette', 'seringue', 'poche',
#              'pilulier', 'ampoule', 'pot', 'stylo', 'film', 'inhalateur',
#              'bouteille', 'vaporateur', 'enveloppe', 'générateur',
#              'boîte', 'aquette', 'sac', 'pompe', 'distributeur',
#              'applicateur', 'fût'
#              ]
element_standard = [x.encode('cp1252') for x in element_standard]


def recode_dosage_lambda1(x):
    try:
        return str(float(x.split()[0]) * 1000) + ' mg'
    except:
        return x

def recode_dosage(table):
    assert 'Dosage' in table.columns
    table = table[table['Dosage'].notnull()].copy()
    table['Dosage'] = table['Dosage'].str.replace(' 000 ', '000 ')
    # il faut le faire 2 fois
    table['Dosage'] = table['Dosage'].str.replace(' 000 ', '000 ')
    table['Dosage'] = table['Dosage'].str.replace('7 500', '7500')
    table['Dosage'] = table['Dosage'].str.replace('4 500', '4500')
    table['Dosage'] = table['Dosage'].str.replace('3 500', '3500')
    table['Dosage'] = table['Dosage'].str.replace('2 500', '2500')
    table['Dosage'] = table['Dosage'].str.replace('1 500', '1500')
    table['Dosage'] = table['Dosage'].str.replace('1 200', '1200')
    table['Dosage'] = table['Dosage'].str.replace('3 700', '3700')
    table['Dosage'] = table['Dosage'].str.replace('O.5', '0.5')  # Faut-il être bête pour mettre un "O" au lieu d'un zéro !
    table['Dosage'] = table['Dosage'].str.replace(',', '.')
    table['Dosage'] = table['Dosage'].str.replace('\. ', '.')
    table['Dosage'] = table['Dosage'].str.replace('µg', 'microgrammes')

    table.loc[table['Dosage'].str.contains(' g'),'Dosage'] = table.loc[table['Dosage'].str.contains(' g'),'Dosage'].apply(lambda x: recode_dosage_lambda1(x))
    table.loc[table['Dosage'].str.contains(' microgrammes'),'Dosage'] = table.loc[table['Dosage'].str.contains(' microgrammes'),'Dosage'].apply(lambda x: str(float(x.split()[0]) / 1000) + ' mg')

    table['Dosage'] = table['Dosage'].str.replace('935.mg', '935 mg')
    table['Dosage'] = table['Dosage'].str.replace('25mg', '25 mg')
    table['Dosage'] = table['Dosage'].str.replace('50mg', '50 mg')

#    table['Dosage'] = table['Dosage'].str.replace('UI', 'U')
    return table

def recode_prix(table):
    assert 'Prix' in table.columns
    table['Prix'] = table['Prix'].str.replace(',','.')
    #Enlever le premier point pour ceux qui en ont deux
    table.loc[table['Prix'].apply(lambda x: x.count('.')) > 1,'Prix'] = table.loc[table['Prix'].apply(lambda x: x.count('.'))>1,'Prix'].str.replace('.','',1)
    table['Prix'] = table['Prix'].apply(lambda x: float(x))
    return table

def recode_ref_dosage(table):
    # TODO: on a des problème de ref dosage.
    assert 'Ref_Dosage' in table.columns
    table = table[table['Ref_Dosage'].notnull()].copy()
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('un ','')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('une ','')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('1ml','1 ml')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('L','l')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace("\(s\)",'')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace(',','.')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('100. 0 g','100.0 g')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('00ml','00 ml')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('1g','1 g')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('1ml','1 ml')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('ml</p>', 'ml')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('comrpimé','comprimé ')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('comprimer','comprimé ')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('comprimé.','comprimé ')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('comprimpé','comprimé ')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('gelule','gélule')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('gélulle','gélule')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('gramme','g')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('récipent','récipient')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('pré-remplie' ,'préremplie')
    table['Ref_Dosage'] = table['Ref_Dosage'].str.replace('sachet dose', 'sachet-dose')
    table['Ref_Dosage'] = table['Ref_Dosage'].apply(recode_litre_en_ml)
    return table


def recode_nom_substance_lambda(string):
    parentesis = re.findall('\(.+\)', string)
    if len(parentesis) == 1:
        parentesis = parentesis[0]
        string = string.replace(parentesis, '')
        parentesis = parentesis[1:]
        parentesis = parentesis[:-1]
        string = parentesis + ' ' + string
        string = string[:-1]
    return string


def recode_nom_substance(table):
    assert 'Nom_Substance' in table.columns
    table['Nom_Substance'] = table['Nom_Substance'].apply(recode_nom_substance_lambda)
    return table

def recode_PVC(chaine):
    if 'PVC' in chaine:
        try:
            int(chaine[-1])
            return chaine + ' comprimé'
        except:
            pass
    return chaine


def recode_litre_en_ml(chaine):
    chaine = chaine.replace('litres', 'litre')
    if chaine[-2:] == ' l':
        chaine = chaine + 'itre'
    if chaine[:5] == 'litre':
        chaine = '1 ' + chaine
    chaine = chaine.replace(' l ', ' litre ')
    if ' litre' in chaine:
        mots = chaine.split()
        idx_avant = mots.index('litre') - 1
        nombre = mots[idx_avant]
        try:
            nombre = float(nombre)
            nombre *= 1000
            mots[idx_avant] = str(nombre)
            chaine = ' '.join(mots)
        except:
            assert nombre == 'par'
        chaine = chaine.replace(' litre', ' ml')
        return chaine
    else:
        return chaine


def recode_label_presta(table):
    assert 'Label_presta' in table.columns
    # TODO: identifier d'où viennent les label nuls
    table = table[table['Label_presta'].notnull()].copy()
    table['Label_presta'] = table['Label_presta'].str.replace(',', '.')
    table['Label_presta'] = table['Label_presta'].str. \
        replace("\(s\)", '')
    # 1 seul cas, qui n'est pas grave car flacon
    table['Label_presta'] = table['Label_presta'].str.replace('1 1', '1')
    table['Label_presta'] = table['Label_presta'].str.replace('00 00', '0000')
    table['Label_presta'] = table['Label_presta'].str.replace('  l', ' l')

    table['Label_presta'] = table['Label_presta'].apply(recode_litre_en_ml)
    table['Label_presta'] = table['Label_presta'].apply(recode_PVC)
    # un oubli du nombre de comprimé
    table['Label_presta'] = table['Label_presta'].str.replace(
        'plaquette thermoformée PVC polyéthylène PVDC aluminium comprimé',
        'plaquette thermoformée PVC polyéthylène PVDC aluminium 60 comprimé')
#    table['Label_presta'] = table['Label_presta'].str.replace('1 kg', '1000 g')
#    table['Label_presta'] = table['Label_presta'].str.replace('litres', 'litre')
#    table['Label_presta'] = table['Label_presta'].str.replace('5.0 l', '5000 ml')
#    table['Label_presta'] = table['Label_presta'].str.replace('2 l ', '2000 ml ')
#    table['Label_presta'] = table['Label_presta'].str.replace('\.5 l ', '500 ml ')
#    table['Label_presta'] = table['Label_presta'].str.replace('0.25 l ', '250 ml ')
#    # on a un cas avec des 2l à la fin
#    table['Label_presta'] = table['Label_presta'].str.replace('de 2 l', ' de 2000 ml')
#    table['Label_presta'] = table['Label_presta'].str.replace('1 fût polyéthylène de 5 l', '1 fût polyéthylène de 5000 ml')

    return table


def extract_quantity(label, reference):

    # TODO: douteux quand la référence apparait plusieurs fois
    # on ne garde que la partie avant la référence
    label = label[:label.index(reference)]
    # s'il y a un "et" ou un " - ", on ne prend que
    # la partie qui concerne la référence
    if " et " in label:
        label = label.split(' et ')[-1]
    if " - " in label:
        label = label.split(" - ")[-1]
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", label)
    floats = [float(x) for x in floats]
    if len(floats) == 0:
        return 1
    return reduce(lambda x, y: x*y, floats)
#    except:
#        print label, reference
#        print row
#        pdb.set_trace()
#        pass

def table_update(table):

    nb_ref_in_label = np.zeros(len(table))
    incoherence_identifiee = []
    reconstitu = []
    i = -1
    for k, row in table[['Ref_Dosage', 'Dosage', 'Label_presta']].iterrows():
        # travail de base sur la référence
        i += 1

        #if i % 100 == 0:
            #print(" on en a fait " + str(i) )
        reference = row['Ref_Dosage']

        if not pd.isnull(reference):
            if reference[:2] == '1 ':
                reference = reference[1:]  # on laisse un espace parce que
                # si ça commence par 1 g, comme ça, ça passe le test avec unit
            ref_floats = re.findall(r"[-+]?\d*\.\d+|\d+", reference)
            #unite = re.search("^" + ref_floats + ": (\w+)", reference)
            if len(ref_floats) > 0:
                ref_floats = [float(x) for x in ref_floats]
                reference_dose = reduce(lambda x, y: x*y, ref_floats) # On multiplie tous les éléments de ref_float ensemble
            else:
                reference_dose = 1
            if reference_dose == 0:
                #un seul cas
                reference_dose = 1
            # travail de base sur le label
            label = row['Label_presta']

            if not isinstance(label, str):
                nb_ref_in_label[i] = np.nan
            else:
                if label.split()[0] in element_standard:
                    label = '1 ' + label

                if reference in label:
                    # TODO: douteux quand la référence apparait plusieurs fois
                    label_dose = extract_quantity(label, reference)
                    nb_ref_in_label[i] = label_dose/reference_dose

                if nb_ref_in_label[i] == 0:
                    for unite in ['ml', 'l', 'mg', 'g', 'dose', 'litre']:
                        if len(reference) >= len(unite) + 1:
                            if ' ' + unite + ' ' in reference or \
                               reference[-(len(unite) + 1):] == ' ' + unite or \
                               reference[:len(unite)] == unite :
                                if ' ' + unite in label:
                                    nb_ref_in_label[i] = extract_quantity(label, ' ' + unite)/reference_dose

                if nb_ref_in_label[i] == 0:
                    reference = row['Ref_Dosage']
                    contenant = [var for var in element_standard
                                 if var in reference]
                    if len(contenant) == 1:
                        var = contenant[0]
                        if var in label:
                            label_dose = extract_quantity(label, var)
                            nb_ref_in_label[i] = label_dose

                if nb_ref_in_label[i] == 0:
                    reference = row['Ref_Dosage']
                    if reference in ['lyophilisat', '1 flacon', 'dose mesurée']:
                        nb_ref_in_label[i] = extract_quantity(label, 'flacon')

                if nb_ref_in_label[i] == 0:
                    reference = row['Ref_Dosage']
                    if ((any(masse in reference for masse in ['g', 'mg']) and
                        any(vol in label for vol in ['l', 'ml'])) or
                        (any(masse in label for masse in ['g', 'mg']) and
                        any(vol in reference for vol in ['l', 'ml']))):
                        incoherence_identifiee += [i]

                    elif reference == 'comprimé' and 'gélule' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'gélule')
                    elif 'qsp' in row['Dosage']:
                        incoherence_identifiee += [i]
                    elif reference == 'pression':
                        incoherence_identifiee += [i]
                    elif 'Bq' in label:  # GBq, MBq
                        pass
                    elif 'Bq' in row['Dosage']:  # GBq, MBq
                        pass
                    elif reference == 'dose':
                        # TODO:
                        pass
                    elif reference == 'sachet-dose' and 'sachet' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'sachet')
                    elif reference == 'flacon de lyophilisat' and 'flacon' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'flacon')
                    elif reference in ['ampoule ou flacon', 'flacon ou ampoule'] and 'flacon' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'flacon')
                    elif reference in ['ampoule ou flacon', 'flacon ou ampoule'] and 'ampoule' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'ampoule')
                    elif reference == 'ampoule de lyophilisat' and 'ampoule' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'ampoule')
                    elif reference == 'emplâtre' and 'sachet' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'sachet')
                    elif reference == 'dispositif cutané' and 'sachet' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'sachet')
                    elif reference == 'flacon' and 'récipient unidose' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'récipient')
                    elif reference == 'flacon' and 'ampoule' in label:
                        nb_ref_in_label[i] = extract_quantity(label, 'ampoule')
                    elif reference == '1 ml de solution reconstituée':
                        reconstitu += [i]
                    elif 'sachet-dose n' in reference:
                        pass
                        # TODO: un truc avec les sachets-doses numérotés
                    elif 'cm^2' in reference:
                        pass
                        # TODO: un truc avec les sachets-doses numérotés
                    else:
                        pass
    return nb_ref_in_label
#            print('il faut tenter autre chose')
#            print(row)
#            pdb.set_trace()

def table_SMR(maj_bdm=maj_bdm):
    def _load_data_SMR(name):
        path = os.path.join(path_gouv, maj_bdm, 'CIS_HAS_' + name + '_bdpm.txt')
        tab = pd.read_table(path, header=None)
        tab.columns = dico_variables['HAS_SMR_bdpm']
#        tab.drop(['HAS','Libelle_SMR'], axis=1, inplace=True)
        return tab

    tab1 = _load_data_SMR('SMR')
    tab2 = _load_data_SMR('ASMR')
    dico_rename = dict(I='Majeur',
                       II='Important',
                       III='Modéré',
                       IV='Mineur',
                       V='Inexistant')

    print(dico_rename)
    tab2['Valeur_SMR'].replace(dico_rename, inplace=True)
    test = tab1.merge(tab2, on=['CIS','Date_SMR'])
    tab1[(tab1['CIS'] == '60529136')]
    import pdb
    pdb.set_trace()



def load_medic_gouv(maj_bdm=maj_bdm, var_to_keep=None, CIP_not_null=False):
    ''' renvoie les tables fusionnées issues medicament.gouv.fr
        si var_to_keep est rempli, on ne revoit que la liste des variables
    '''
    # chargement des données
    output = None
    for name, vars in dico_variables.items():
        # teste si on doit ouvrir la table
        if var_to_keep is None:
            intersect = vars
        if var_to_keep is not None:
            intersect = [var for var in vars if var in var_to_keep]
        if len(intersect) > 0:
            path = os.path.join(path_gouv, maj_bdm, 'CIS_' + name + '.txt')
            tab = pd.read_table(path, header=None, encoding='cp1252')
            if name in ['COMPO_bdpm', 'GENER_bdpm']:
                tab = tab.iloc[:, :-1]
            tab.columns = vars
            if name in ['HAS_ASMR_bdpm','HAS_SMR_bdpm']:
                #On ne selectionne que les médicaments pour lesquels on a un CIS sans lettres (normal)
                tab = tab.loc[tab['CIS'].apply(lambda x: len(re.findall("[A-Za-z]", x)))==0,:]
                tab['CIS'] = tab['CIS'].apply(lambda x: float(x))
                if 'Date_ASMR' in vars:
                    tab['Date_ASMR'] = tab['Date_ASMR'].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d").date())
                # max ASMR = 13
            tab = tab[['CIS'] + intersect]
            # correction ad-hoc...
            if tab['CIS'].dtype == 'object':
                problemes = tab['CIS'].str.contains('REP', na=False)
                problemes = problemes | tab['CIS'].isin(['I6049513', 'inc     '])
                tab = tab.loc[~problemes, :]
                tab['CIS'].astype(int)

            if 'CIP' in intersect:
                tab['CIP'] = tab['CIP'].astype(str)
#            if 'Ref_Dosage' in intersect:
#                tab = recode_ref_dosage(tab)
#            if 'Dosage' in intersect:
#                tab = recode_dosage(tab)
#            if 'Label_presta' in intersect:
#                tab = recode_label_presta(tab)
#            if 'Prix' in intersect:
#                tab = recode_prix(tab)
#            if 'Nom_Substance' in intersect:
#                tab = recode_nom_substance(tab)
            if output is None:
                output = tab
                print("la première table est " + name + " , son nombre de " +
                      "lignes est " + str(len(output)))
            else:

                output = output.merge(tab, how='outer', on='CIS',
                                      suffixes=('', name[:-4]))
                if CIP_not_null:
                    if 'CIP' in output.columns:
                        output = output[output['CIP'].notnull()]
                print("après la fusion avec " + name + " la base mesure " +
                      str(len(output)))

    # On met les dates au format datetime
    for var in var_to_keep:
        if 'date' in var or 'Date' in var:
#            print('On retire ' + str(sum(output[var].isnull())) + " valeurs parce " +
#                   "qu'il n'y a pas de date")
            sel = output[var].notnull()
            if var not in ['Date_ASMR', 'Date_SMR']:
                output.loc[sel, var]  = output.loc[sel, var].map(lambda t : dt.datetime.strptime(t, "%d/%m/%Y").date())
                for time_idx in ['month', 'year']:
                    name = var + '_' + time_idx
                    output[name] = 0
                    output[name][output[var].notnull()] = output[var][output[var].notnull()].apply(lambda x: getattr(x, time_idx))

    if 'nb_ref_in_label_medic_gouv' in var_to_keep:
        output['nb_ref_in_label_medic_gouv'] = table_update(output)
#    if 'mode_prise' in var_to_keep:
#        output['mode_prise'] = mode_prise(output)

    return output

if __name__ == '__main__':
#table = load_medic_gouv(maj_bdm, Z['Etat','Date_AMM','CIP7','Label_presta','Date_declar_commerc','Taux_rembours','Prix','Id_Groupe','Type',
#                                  'indic_droit_rembours', 'Statu_admin_presta','Element_Pharma','Code_Substance','Nom_Substance','',
#                                  'Ref_Dosage','Nature_Composant','Substance_Fraction'])
#     test = load_medic_gouv(maj_bdm)

#    table = load_medic_gouv(maj_bdm, ['CIP7', 'Label_presta',
#                                      'Element_Pharma','Code_Substance','Nom_Substance','Dosage',
#                                      'Ref_Dosage','Nature_Composant','Substance_Fraction'])

    info_utiles_from_gouv = ['CIP7', 'CIP', 'Nom', 'Id_Groupe', 'Prix', 'Titulaires',
                             'Num_Europe',
                             'Element_Pharma', 'Code_Substance', 'Nom_Substance',
                             #'Nature_Composant', 'Substance_Fraction',
                             'Libelle_ASMR', 'Type', 'Ref_Dosage', 'Dosage',
                             'Date_declar_commerc', 'Date_AMM', 'Taux_rembours',
                             'indic_droit_rembours', 'Statu_admin_presta',
                             'Label_presta','Valeur_ASMR', 'Date_ASMR',
                             'Label_presta','Valeur_SMR', 'Date_SMR',
                             #'nb_ref_in_label_medic_gouv',
                             'premiere_vente', 'derniere_vente']

    table = load_medic_gouv(maj_bdm, info_utiles_from_gouv)



    table = table[~table['Element_Pharma'].isin(['pansement', 'gaz'])]

    for var in ['Ref_Dosage', 'Dosage', 'Label_presta']:
        print(table[var].isnull().sum())
        table = table[table[var].notnull()]


#HAS_SMR_bdpm=['CIS', 'HAS', 'Evalu', 'Date', 'Valeur_SMR', 'Libelle_SMR']