# run_analysis.py
import sys
import awkward as ak 
import dask
from coffea import util
from coffea.nanoevents import NanoAODSchema
from dask.diagnostics import ProgressBar
from coffea.dataset_tools import preprocess, apply_to_fileset, max_chunks
from dataset import get_fileset
from distributed import Client  
import os
import h5py
import importlib
from datetime import datetime
import argparse
from region_abcd_proc import ElectronChannel

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run ABCD method for QCD estimation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_analysis.py  --year 2017 --region A --btagWP M --pileUpWP M
  python3 run_analysis.py  --year 2018 --region B --btagWP M --pileUpWP M 
        """
    )

    parser.add_argument(
        '--year',
        type=str,
        required=True,
        choices=['2016postVFP','2016preVFP','2017','2018'],
        help='Data-taking eras/year'
    )

    parser.add_argument(
        '--region',
        type=str,
        required=True,
        choices=['A','B','C','D'],
        help='Region for ABCD'        
    )

    parser.add_argument(
        '--btagWP',
        type=str,
        required=True,
        choices=['L','M','T'],
        help='BTagging working point'
    )

    parser.add_argument(
        '--pileUpWP',
        type=str,
        required=True,
        choices=['L','M','T'],
        help='PileUp working point'
    )
    
    parser.add_argument(
        '--jetpt',
        type=float,
        required=True,
        default =20.0,
        help='Minimum Jet pt'
    )
    
    parser.add_argument(
        '--choice',
        type=int,
        required=True,
        choices=[1,2],
        help='1-->ID VS MT , 2-->ID VS NBTAGS'
    )
    return parser.parse_args()

def main():
    #Parse command line arguments
    args = parse_args()

    print("="*60)
    print(f"Welcome to ABCD Method, Here we shall estimate the QCD !!!")
    print("="*60)
    print(f"Configuration")
    print(f"  Era : {args.year}")
    print(f"  Region : {args.region}")
    print(f"  Btagging_wp :{args.btagWP} ")
    print(f"  PileUp_wp:{args.pileUpWP}")
    print(f"  Minimum Jet Pt:{args.jetpt}")
    print(f"  ABCD_CHOICE:{args.choice}")
    print("="*60)


    #Initialize processor with command line arguments
    processor_instance = ElectronChannel(year=args.year , region = args.region ,btagWP = args.btagWP ,pileUpWP = args.pileUpWP , jetPt=args.jetpt , choice = args.choice)

    client =Client()

    print("-----Fileset Loading---")
    fileset = get_fileset()
    
    print(f"\n DataSet summary for {args.year}:")
    print("-"*60)
    for key, info in fileset.items():
        print(f"{key:<20} : {len(info['files']):>4} files")
    

    print(f"\n{'='*60}")
    print(f"Hi! I am running analysis for the Region {args.region}, Year {args.year}....")
    print(f"{'='*60}\n")

    # Process the data
    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        files_per_batch=1,
        skip_bad_files=True,
        save_form=False,
    )
    
    to_compute = apply_to_fileset(
        processor_instance,
        max_chunks(dataset_runnable, 300),
        schemaclass=NanoAODSchema,
    )
    
    with ProgressBar():
        (out,) = dask.compute(to_compute, scheduler='threads')
    
    output_dir = "/mnt/disk2/sanskar/2016preVFP_RECO/RECO_UPDATED_MUKUND"
    os.makedirs(output_dir, exist_ok = True)

    for dataset, nested in out.items():
        output_filename = os.path.join(output_dir, f"{dataset}_reco_variables_updated.parquet")

        reco_features = nested[dataset]
        data_to_save = ak.Array(reco_features)
        ak.to_parquet(data_to_save, output_filename, compression="GZIP")

    print("The reco features have been saved successfully!")
    print(f"The results were saved to {output_dir}")
                                                    
    '''
    print("out done")

    dataset_name = list(out.keys())[0]
    level1 = out[dataset_name]
    features = level1[dataset_name]

    x_m = features["x_m"]
    res = features["res"]
    perm_idx = features["perm_idx"]
    b_lep_mass = features["b_lep_mass"]
    b_had_mass = features["b_had_mass"]
    l1_mass = features["l1_mass"]
    l2_mass = features["l2_mass"]
    charge = features["charge"]
    gen_top_mass = features["gen_top_mass"]
    gen_antitop_mass = features["gen_antitop_mass"]
    gen_top_pt = features["gen_top_pt"]
    gen_antitop_pt = features["gen_antitop_pt"]
    gen_top_rap = features["gen_top_rap"]
    gen_antitop_rap = features["gen_antitop_rap"]
    #gen_top_phi = features["gen_top_phi"]
    #gen_antitop_phi = features["gen_antitop_phi"]
    
    x_m_np = ak.to_numpy(x_m)
    res_np = ak.to_numpy(res)
    perm_idx_np = ak.to_numpy(perm_idx)
    b_lep_mass_np = ak.to_numpy(b_lep_mass)
    b_had_mass_np = ak.to_numpy(b_had_mass)
    l1_mass_np = ak.to_numpy(l1_mass)
    l2_mass_np = ak.to_numpy(l2_mass)
    el_charge_np = ak.to_numpy(charge)
    gen_top_mass_np = ak.to_numpy(gen_top_mass)
    gen_antitop_mass_np = ak.to_numpy(gen_antitop_mass)
    gen_top_pt_np = ak.to_numpy(gen_top_pt)
    gen_top_rap_np = ak.to_numpy(gen_top_rap)
    gen_antitop_pt_np = ak.to_numpy(gen_antitop_pt)
    gen_antitop_rap_np = ak.to_numpy(gen_antitop_rap)
    #gen_antitop_phi_np = ak.to_numpy(gen_antitop_phi)

    
    print("numpy_conversion done")
    #output_dir = "/mnt/disk2/sanskar/2018_BDT/RECO_results"
    output_dir = "/home/irfan/sanskar/outputs/2018_reco/"
    os.makedirs(output_dir, exist_ok = True)

    output_file = os.path.join(output_dir, "ttbar_semilep_reco.h5")
    
    
    datasets = {
        "x_m": x_m_np,
        "res": res_np,
        "perm_idx": perm_idx_np,
        "b_lep_mass": b_lep_mass_np,
        "b_had_mass": b_had_mass_np,
        "l1_mass": l1_mass_np,
        "l2_mass": l2_mass_np,
        "charge":el_charge_np,
        "gen_top_mass":gen_top_mass_np,
        "gen_antitop_mass":gen_antitop_mass_np,
        "gen_top_pt":gen_top_pt_np,
        "gen_antitop_pt":gen_antitop_pt_np,
        "gen_top_rap":gen_top_rap_np,
        "gen_antitop_rap":gen_antitop_rap_np,
        #"gen_top_phi": gen_top_phi_np,
        #"gen_antitop_phi": gen_antitop_phi_np
    }
    


    with h5py.File(output_file, "w") as f:

        #def write(name, ak_array):
            #np_array = ak.to_numpy(ak_array)

        for name, data in datasets.items():
            f.create_dataset(name, data=data, compression="gzip", compression_opts=4)
            #del np.array
        
        
        #write("x_m",x_m)
        #write("res",res)
        #write("perm_idx",perm_idx)
        #write("b_lep_mass",b_lep_mass)
        #write("b_had_mass",b_had_mass)
        #write("l1_mass",l1_mass)
        #write("l2_mass",l2_mass)
        #write("charge",charge)
        #write("gen_top_mass",gen_top_mass)
        #write("gen_antitop_mass",gen_antitop_mass)
        
        
    print(f"✓ Analysis completed successfully!")
    #print(f"Output saved to: {output_path}")
    '''
if __name__ == "__main__":
    main()
