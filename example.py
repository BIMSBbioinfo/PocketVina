from pocketvina import PocketVinaGPU, PocketVinaConfig, P2RankConfig

if __name__ == '__main__':
    config = PocketVinaConfig(
        protein_paths="example/protein_files.txt",
        smiles_file="example/ligands.txt",
        output_dir="results",
        batch_size=10,
        p2rank_config=P2RankConfig(
            threads=4,
            visualizations=False,
            vis_copy_proteins=False,
            output_dir="results"
        )
    )
    pocketvina = PocketVinaGPU(config, htvs_mode=True)
    pocketvina.execute_docking()
