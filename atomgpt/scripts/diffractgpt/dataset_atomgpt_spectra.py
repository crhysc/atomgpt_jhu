import numpy as np
from atomgpt.inverse_models.utils import smooth_xrd
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from tqdm import tqdm

d = data("dft_3d")
# d = data('alex_pbe_hull')

f = open("id_prop.csv", "w")
count = 0
for i in tqdm(d, total=len(d)):
    # if count<10:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = "POSCAR-" + jid + ".vasp"
    atoms.write_poscar(poscar_name)
    y_new_str, cccc = smooth_xrd(atoms=atoms, intvl=0.3, thetas=[0, 90])
    f.write("%s,%s\n" % (poscar_name, y_new_str))
    count += 1
    # if count == max_samples:
    # break
f.close()
