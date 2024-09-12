from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
from collections import defaultdict
from rdkit.Chem import rdFingerprintGenerator
import cairosvg
import numpy as np
import sys
import io
from PIL import Image

def map_to_color(value):
    # Assicurati che il valore sia nell'intervallo [-100, 100]
    value = max(min(value, 100), -100)
    
    # Mappa il valore all'intervallo [0, 1]
    normalized = (value + 100) / 200
    
    # Calcola i componenti RGB
    if value < 0:
        # Interpola tra blu (0, 0, 255) e bianco (255, 255, 255)
        blue = 255
        red = green = int((1 + value / 100) * 255)
    else:
        # Interpola tra bianco (255, 255, 255) e rosso (255, 0, 0)
        red = 255
        green = blue = int((1 - value / 100) * 255)
    
    return (red/255.0, green/255.0, blue/255.0)

def map_to_color_x(value,x):
    # Assicurati che il valore sia nell'intervallo [-100, 100]
    value = max(min(value, x), -x)
    
    # Calcola i componenti RGB
    if value < 0:
        # Interpola tra blu (0, 0, 255) e bianco (255, 255, 255)
        red = 255
        blue = green = int((1 + value / x) * 255)
    else:
        # Interpola tra bianco (255, 255, 255) e rosso (255, 0, 0)
        blue = 255
        green = red = int((1 - value / x) * 255)
    
    return (red/255.0, green/255.0, blue/255.0)

def map_to_color_ok(value, v_min, v_max):
    # Assicurati che il valore sia nell'intervallo [v_min, v_max]
    value = max(min(value, v_max), v_min)
    
    # Mappa il valore all'intervallo [0, 1]
    normalized = (value - v_min) / (v_max - v_min)
    
    # Calcola i componenti RGB
    if normalized < 0.5:
        # Interpola tra blu (0, 0, 255) e bianco (255, 255, 255)
        blue = 255
        red = green = int((normalized * 2) * 255)
    else:
        # Interpola tra bianco (255, 255, 255) e rosso (255, 0, 0)
        red = 255
        green = blue = int((2 - normalized * 2) * 255)
    
    return (red/255.0, green/255.0, blue/255.0)

def get_atoms_within_radius(mol, central_atom_idx, radius):
    # Calcola la matrice delle distanze
    distance_matrix = Chem.GetDistanceMatrix(mol)
    
    # Trova gli atomi entro il raggio desiderato
    atoms_within_radius = np.where(distance_matrix[central_atom_idx] <= radius)[0]
    
    return list(atoms_within_radius)

f = open(sys.argv[1],"r")
data = f.read().split("\n")
f.close()

smiles = data[1].split(",")[0].replace("\"","")
m = Chem.MolFromSmiles(smiles)
Chem.RemoveStereochemistry(m)

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateAtomCounts()
ao.AllocateAtomToBits()
ao.AllocateBitInfoMap()

w = []
vals = data[3:-1]
for i in range(len(vals)):
    w.append(float(vals[i].split(",")[1]))

fp = fpgen.GetFingerprint(m,additionalOutput=ao)
print(fp.ToBitString())
info = ao.GetBitInfoMap()
print(info)

w_atom = []
c_atom = []
for i in range(len(m.GetAtoms())):
    w_atom.append(0.0)
    c_atom.append(0.0)

w_tot1 = 0.0
w_tottot = 0.0
for i in range(len(w)):
    w_tottot = w_tottot + w[i]
    if i in info.keys():
        w_tot1 = w_tot1 + w[i]
        for idx,radius in info[i]:
            atom_indices = get_atoms_within_radius(m, idx, radius)
            for j in atom_indices:
                w_atom[j]=w_atom[j]+w[i]
                c_atom[j]=c_atom[j]+1.0

print("TOTAL="+str(w_tottot))
print("SOMMMMMAAAAA="+str(w_tot1))
#for i in range(len(w_atom)):
#    w_atom[i]=w_atom[i]/c_atom[i]

drawer = rdMolDraw2D.MolDraw2DSVG(1200, 1200)
atoms = list(range(len(m.GetAtoms())))
highlight_atom_map = defaultdict(list)
highlight_bond_map = defaultdict(list)
highlight_radii = defaultdict(float)

#w_atom = np.ones(len(w_atom))*-100

www = np.array(w_atom)
mean_www = np.mean(abs(www))
std_www = np.std(abs(www))
#for i in range(len(www)):
#    if (abs(www[i])<mean_www):
#        w_atom[i]=0.0
cap = mean_www + 3.0*std_www

############schifezza
#legami = []
#for legame in m.GetBonds():
#    indice_atomo_inizio = legame.GetBeginAtomIdx()
#    indice_atomo_fine = legame.GetEndAtomIdx()
#    legami.append((indice_atomo_inizio, indice_atomo_fine))

# Stampa la lista dei legami
#for legame in legami:
#    print(f"Legame tra atomo {legame[0]} e atomo {legame[1]}")
#################fine schifezza

for atom in atoms:
        #highlight_atom_map[atom].append(map_to_color_ok(w_atom[atom],min(w_atom),max(w_atom)))
        #highlight_bond_map[atom].append(map_to_color_ok(w_atom[atom],min(w_atom),max(w_atom)))
        highlight_atom_map[atom].append(map_to_color_x(w_atom[atom],cap))
        #highlight_bond_map[atom].append(map_to_color_x(w_atom[atom],cap))
        highlight_radii[atom] = 0.6

#print (len(highlight_atom_map))
#print (len(highlight_bond_map))

drawer.DrawMoleculeWithHighlights(m, "",  highlight_atom_map=dict(highlight_atom_map),
                                   highlight_bond_map=dict(highlight_bond_map),
                                   highlight_radii=dict(highlight_radii),
                                   highlight_linewidth_multipliers={})
drawer.FinishDrawing()
svg = drawer.GetDrawingText()

png_filename = sys.argv[1].split('.')[0] + '.png'
cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_filename, dpi=600)

# Salva il file TIFF
#png_image = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
#image = Image.open(io.BytesIO(png_image))
#tiff_filename = sys.argv[1].split('.')[0] + '.tiff'
#image.save(tiff_filename, format='TIFF',dpi=[300,300])
