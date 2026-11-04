"""Look up the PocketVina Atlas without docking (no GPU needed): python example/atlas.py

It reads the public Atlas over the internet, or the folder or URL in $POCKETVINA_ATLAS. The files it writes go to the
working folder. More: https://bioinformatics.mdc-berlin.de/PocketVina/atlas.html
"""
from pocketvina import Atlas

atlas = Atlas()                                        # the public Atlas
print(atlas.search("imatinib", top=5))                 # proteins and drugs that match, best first
print(atlas.best_scores(protein="ABL1", top=10))       # the drugs with the best Vina score on ABL1
print(atlas.best_scores(ligand="imatinib", top=10))    # the proteins imatinib scores best on

pair = atlas.select(protein="ABL1", ligand="imatinib", pocket="all")  # the best pose in every pocket
print(pair.show(path="ABL1_imatinib.html"))            # the 3D page: open this file in a browser
print(pair.export())                                   # the poses as an SDF file
print(pair.export_protein())                           # the protein (its AlphaFold model) as a PDB file

atlas.download(proteins=["ABL1"], out="my_atlas")      # ABL1's files, to work offline (it says the size first)
print(Atlas("my_atlas").best_scores(protein="ABL1", top=5))  # read from the folder, without internet
