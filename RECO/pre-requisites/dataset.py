import glob
import os

def get_fileset():
    # Helper function: converts globbed paths into {"path": "Events"} format
    def dictionary(path_pattern):
        files = glob.glob(path_pattern, recursive=True)
        if not files:
            print(f"Warning: No files found for pattern: {path_pattern}")
        return {"files":{fp: "Events" for fp in files}}
        
    # Base directory for all datasets
    base_dir = "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/MC"
    
    # Dataset configuration: name -> relative path pattern
    dataset_config = {
        "ttbar_SemiLeptonic": "ttbar_SemiLeptonic/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/Tree_24_Nov23_MCUL2016preVFP_ttbar_SemiLeptonic/231124_104407/0000/*.root",
        #"Tchannel": "Tchannel/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/Tree_24_Nov23_MCUL2016preVFP_Tchannel/231124_105450/0000/*.root",
        #"Schannel": "Schannel/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/Tree_22_Nov23_MCUL2016preVFP_Schannel/231122_155433/0000/*.root",
        #"ttbar_FullyLeptonic": "ttbar_FullyLeptonic/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/Tree_27_Nov23_MCUL2016preVFP_ttbar_FullyLeptonic/231127_092402/0000/*.root",
        #"tw_top": "tw_top/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/Tree_24_Nov23_MCUL2016preVFP_tw_top/231124_105024/0000/*.root",
        #"tw_antitop": "tw_antitop/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/Tree_22_Nov23_MCUL2016preVFP_tw_antitop/231122_125756/0000/*.root",
        #"DYJetsToLL": "DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Tree_24_Nov23_MCUL2016preVFP_DYJetsToLL/231124_105320/0000/*.root",
        #"QCD_Pt-30to50EMEnriched":"QCD_Pt-30to50_EMEnriched/QCD_Pt-30to50_EMEnriched_TuneCP5_13TeV-pythia8/Tree_24_Nov23_MCUL2016preVFP_QCD_Pt-30to50_EMEnriched/231124_105629/0000/*.root",
        #"QCD_Pt-50to80EMEnriched":"QCD_Pt-50to80_EMEnriched/QCD_Pt-50to80_EMEnriched_TuneCP5_13TeV-pythia8/Tree_22_Nov23_MCUL2016preVFP_QCD_Pt-50to80_EMEnriched/231122_155316/0000/*.root",
        #"QCD_Pt-80to120EMEnriched":"QCD_Pt-80to120_EMEnriched/QCD_Pt-80to120_EMEnriched_TuneCP5_13TeV-pythia8/Tree_27_Nov23_MCUL2016preVFP_QCD_Pt-80to120_EMEnriched/231127_091756/0000/*.root",
        #"QCD_Pt-120to170EMEnriched":"QCD_Pt-120to170_EMEnriched/QCD_Pt-120to170_EMEnriched_TuneCP5_13TeV-pythia8/Tree_28_Nov23_MCUL2016preVFP_QCD_Pt-120to170_EMEnriched/231128_064314/0000/*.root",
        #"QCD_Pt-170to300EMEnriched":"QCD_Pt-170to300_EMEnriched/QCD_Pt-170to300_EMEnriched_TuneCP5_13TeV-pythia8/Tree_28_Nov23_MCUL2016preVFP_QCD_Pt-170to300_EMEnriched/231128_064733/0000/*.root",
        #"QCD_Pt-300toInfEMEnriched":"QCD_Pt-300toInf_EMEnriched/QCD_Pt-300toInf_EMEnriched_TuneCP5_13TeV-pythia8/Tree_28_Nov23_MCUL2016preVFP_QCD_Pt-300toInf_EMEnriched/231128_064441/0000/*.root",
        #"Tbarchannel": "Tbarchannel/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/Tree_24_Nov23_MCUL2016preVFP_Tbarchannel/231124_105152/0000/*.root",
        #"WJetsToLNu_0J": "WJetsToLNu_0J/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/Tree_22_Nov23_MCUL2016preVFP_WJetsToLNu_0J/231122_155824/0000/*.root",
        #"WJetsToLNu_1J": "WJetsToLNu_1J/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/Tree_22_Nov23_MCUL2016preVFP_WJetsToLNu_1J/231122_155549/0000/*.root",
        #"WJetsToLNu_2J": "WJetsToLNu_2J/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/Tree_22_Nov23_MCUL2016preVFP_WJetsToLNu_2J/231122_155020/0000/*.root",
        #"WWTo2L2Nu": "WWTo2L2Nu/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/Tree_27_Nov23_MCUL2016preVFP_WWTo2L2Nu/231127_092059/0000/*.root",
        #"WWTolnulnu": "WWTolnulnu/WWTolnulnu_TuneCP5_13TeV-madgraph-pythia8/Tree_27_Nov23_MCUL2016preVFP_WWTolnulnu/231127_091656/0000/*.root",
        #"WZTo2Q2L": "WZTo2Q2L/WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/Tree_27_Nov23_MCUL2016preVFP_WZTo2Q2L/231127_091956/0000/*.root",
        #"ZZTo2L2Nu": "ZZTo2L2Nu/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/Tree_24_Nov23_MCUL2016preVFP_ZZTo2L2Nu/231124_104555/0000/*.root",
        #"ZZTo2Q2L": "ZZTo2Q2L/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/Tree_22_Nov23_MCUL2016preVFP_ZZTo2Q2L/231122_155942/0000/*.root",
    }
    
    '''
    data_dir = [
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016B_ver1_el/SingleElectron/Tree_30_Nov23_Run2016B_ver1_el/231130_152542/0000",
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016B_ver2_el/SingleElectron/Tree_30_Nov23_Run2016B_ver2_el/231130_152726/0000",
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016C_HIPM_el/SingleElectron/Tree_30_Nov23_Run2016C_HIPM_el/231130_152352/0000",
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016D_HIPM_el/SingleElectron/Tree_30_Nov23_Run2016D_HIPM_el/231130_152634/0000",
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016E_HIPM_el/SingleElectron/Tree_30_Nov23_Run2016E_HIPM_el/231130_151922/0000",
        "/nfs/home/common/RUN2_UL/Tree_crab/SIXTEEN_preVFP/Data_el/Run2016F_HIPM_el/SingleElectron/Tree_30_Nov23_Run2016F_HIPM_el/231130_152445/0000"
    ]
    
    datafiles = []
    
    for d in data_dir:
        datafiles.extend(glob.glob(d + "/*.root"))
    '''

    # Build full fileset
    fileset = {name: dictionary(os.path.join(base_dir, path)) for name, path in dataset_config.items()}
    #fileset["DATA"] = {"files":{fp: "Events" for fp in datafiles}}
    
    
    return fileset

