import hist
import dask
import awkward as ak
import hist.dask as hda
import numpy as np
import dask_awkward as dak
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.lookup_tools import extractor
from coffea.dataset_tools import apply_to_fileset, max_chunks, preprocess
import os
from coffea.nanoevents.methods import vector
import correctionlib
from correctionlib import CorrectionSet
import json
from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper
import warnings
import operator
from collections import namedtuple
warnings.filterwarnings("ignore")


class TTbarXGBWrapper(xgboost_wrapper):
    def __init__(self, model_path, feature_list):
        super().__init__(model_path)
        self.feature_list = feature_list

    def prepare_awkward(self, events):

        data = ak.concatenate(
            [events[name][:, None] for name in self.feature_list],
            axis=1
        )

        # Required return format for coffea.ml_tools
        return [], {"data": data}

class ElectronChannel(processor.ProcessorABC):
    def __init__(self, year = "2018" ,region = "D",btagWP="M",pileUpWP = "L",jetPt=20.0, choice=1):
        
        '''
        Enhanced ElectronChannel processor with two ABCD method choices
        
        Parameters:
        ------------
        year : str
            Data Taking year (2016postVFP,2016preVFP ,2017, 2018)
        
        region : str
            ABCD regions (A, B ,C ,D)

        btag WP : str
            B-tagging working point (L ,M ,T)

        pileUp WP : str
            PileUp jet ID working point (L ,M ,T)

        jet_pt : float
            Minimum pt of the jet required in the analysis

        choice : int 
            ABCD method choice:
            --1. Cutbased Id vs mT (requires >=1 btags for all regions)
            --2. Cutbased Id vs nbtags (requires mT >50 for all regions)
        '''
        # ------- Initialise the Model --------------#
        self.bdt_features = ["FW1","Sxz","Szz","AL","p2in","planarity","pT_Sum","nJet","delta_R","dphi_lb"]
        self.bdt = TTbarXGBWrapper("trained_model_for_ttbar.xgb", self.bdt_features)
        
        # -------- Mass of the particles -------------#
        self.MW = 80.4
        self.Mtt = 172.5
        self.sigmaW = 10.0
        self.sigmatt = 13.0

        #-------define all the constants here -------#
        self.region = region
        self.year = year
        self.bTagWP = btagWP
        self.PileUpWP = pileUpWP
        self.choice = choice
        self.JetPt = jetPt
        
        # ----- Validate Inputs ----------#
        self.available_eras = ["2016postVFP", "2016preVFP", "2017" ,"2018"]
        self.available_regions = ["A","B", "C" ,"D"]
        self.available_choices = [1,2]


        if self.year not in self.available_eras:
            raise ValueError(f"Invalid year : {self.year}. Choose from {self.available_eras}")
        if self.region not in self.available_regions:
            raise ValueError(f"Invalid region: {self.region}. Choose from {self.available_regions}")
        if self.choice not in self.available_choices:
            raise ValueError(f"Invalid choice: {self.choice}. Choose from {self.available_choices}")
    
        # ----------CHOICE 1 : CUTBASED ID VS mT (WITH >=1 BTAGS) --------#
        self.region_def_choice1 = {
            "A":{"CutbasedId":1, "wp":"Veto","mT":40,"mT_op":"<","nbtags":1, "nbtags_op":">="},
            "B":{"CutbasedId":4, "wp":"Tight", "mT":40,"mT_op":"<","nbtags":1, "nbtags_op":">="},
            "C":{"CutbasedId":1, "wp":"Veto", "mT":50,"mT_op":">", "nbtags":1, "nbtags_op":">="},
            "D":{"CutbasedId":4, "wp":"Tight","mT":50,"mT_op":">", "nbtags":1, "nbtags_op":">="}
        }
        
        # ----------CHOICE 2 : CUTBASED ID VS NBTAGS (WITH mT > 50)-------#
        self.region_def_choice2 = {
            "A": {"CutbasedId": 1, "wp": "Veto", "mT": 50, "mT_op": ">", "nbtags": 0, "nbtags_op": "=="},
            "B": {"CutbasedId": 4, "wp": "Tight", "mT": 50, "mT_op": ">", "nbtags": 0, "nbtags_op": "=="},
            "C": {"CutbasedId": 1, "wp": "Veto", "mT": 50, "mT_op": ">", "nbtags": 1, "nbtags_op": ">="},
            "D": {"CutbasedId": 4, "wp": "Tight", "mT": 50, "mT_op": ">", "nbtags": 1, "nbtags_op": ">="}
        }
        
        # ----- REGION SELECTION CHOICE -------------------#
        self.RegionConfig = {
            1 : self.region_def_choice1,
            2 : self.region_def_choice2
        }
        
        # ----- DEPLOY THE OPERATION ----------------------#
        self.operations = self.RegionConfig[self.choice][self.region]
        
        self.etaEl_values = {
            "2016postVFP":2.1,
            "2016preVFP": 2.1,
            "2018": 2.4,
            "2017":2.4
        }

        self.HLT_wp = {
            "A":"Veto",
            "B":"Tight",
            "C":"Veto",
            "D":"Tight"
        }
        
        self.HLT_scalefactor = self.HLT_wp[self.region]
        #------for electron--------------#
        self.ElectronPt = 35
        self.ElectronEta = self.etaEl_values[self.year]
        self.CutBasedIdTight = self.operations["CutbasedId"]
        self.IdWP = self.operations["wp"]
        
        #------Barrel and End Cap regions----# 
        self.barrel = 1.444
        self.endcap = 1.566
        
        #------for Jets----------------------#
        self.JetEta = 2.4  
        
        #self.bTagWP = "M"
        self.btag_thresholds_by_year = {
            # Values from: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation
            "2018": {
                "L": 0.0490,
                "M": 0.2783,
                "T": 0.7100
            },
            "2017": {
                "L": 0.0532,
                "M": 0.3040,
                "T": 0.7476
            },
            "2016postVFP": {
                "L": 0.0480,
                "M": 0.2489,
                "T": 0.6377
            },
            "2016preVFP": {
                "L": 0.0508,
                "M": 0.2598,
                "T": 0.6502
            }
        }

        self.DeepJetWP = self.btag_thresholds_by_year[self.year][self.bTagWP]
        
        #-----mT mask------------------------#

        self.mT = self.operations["mT"]
        self.mT_operator = self.operations["mT_op"]
        self.nbtags_threshold = self.operations["nbtags"]
        self.nbtags_operator = self.operations["nbtags_op"]

        #-----Operations on mT(if choosen option =1) and nbtags(if choosen option = 2)--------------#
        self.ops = {
            ">":operator.gt,
            "<":operator.lt,
            ">=":operator.ge,
            "<=":operator.le,
            "==":operator.eq
        }

        #-----Event Selection Requirements------#
        self.nElectron = 0   
        self.nJets = 4
                
        #------Flavor specific Jets----------#
        self.bflav = 5
        self.cflav = 4
        self.lflav = 0
        
        #-------Histogram for Electron-------#
        #----For PT---#
        self.ePtBins = 27
        self.ePtStart = 30
        self.ePtEnd = 300
        
        #---For Eta----#
        self.etaConfig = {
            "2016postVFP":42,
            "2016preVFP":42,
            "2018":48,
            "2017":48
        }


        self.eEtaBins = self.etaConfig[self.year]
        self.eEtaStart= -self.ElectronEta
        self.eEtaEnd = self.ElectronEta
        
        #----For Phi---#
        self.ePhiBins = 20
        self.ePhiStart = -np.pi
        self.ePhiEnd = np.pi
        
        #------Histogram for Jet-------------#
        #----FOR PT--------#
        self.jPtBins = 38
        self.jPtStart= 20
        self.jPtEnd = 400
        
        #----FOR ETA-------#
        self.jEtaBins = 48
        self.jEtaStart = -2.4
        self.jEtaEnd = 2.4
        
        #----FOR PHI-------#
        self.jPhiBins = 20
        self.jPhiStart = -np.pi
        self.jPhiEnd = np.pi
        
        #---Pileup Jet id wp---#

        self.PileUpConfig = {
            "2016postVFP":{
                "L": (1 << 0),
                "M": (1 << 1),
                "T": (1 << 2)
            },
            "2016preVFP":{
                "L": (1 << 0),
                "M": (1 << 1),
                "T": (1 << 2)
            },
            "2017":{
                "L": (1 << 2),
                "M": (1 << 1),
                "T": (1 << 0)
            },
            "2018":{
                "L": (1 << 2),
                "M": (1 << 1),
                "T": (1 << 0)
            }
        
        }
    
        self.res_values = {"electron":0.05,"MET":0.05,"Jet":0.10}
        
        self.BDT_config = {
            "2016postVFP": 0.523,
            "2016preVFP": 0.521,
            "2018":0.52,
            "2017":0.52
        }

        self.BDT_cut_val = self.BDT_config[self.year]
        
    def process(self, events):
        '''
        Process events and apply ABCD method based on selected choice
        '''
        isRealData = not hasattr(events, "GenPart")
        dataset = events.metadata['dataset']
        print("\n")
        print(f"current dataset : {dataset}")
        print("--" * 50)
        
        
        selection = PackedSelection()
        
        #----Print the characteristics of the current region -----#
        print("\n" + "="*60)
        print("ABCD METHOD CONFIGURATION")
        print("="*60)
        print(f"Choice               : {self.choice}")
        if self.choice == 1:
            print(f"Method               : CutBasedID vs mT")
            print(f"B-tag requirement    : >= 1 for all regions")
            print(f"================FLOWCHART=========================")
            print("                      ID =Loose       ID = Tight")
            print("                  ┌─────────────────┬─────────────────┐")
            print("    mT < 40       │   Region A      │   Region B      │")
            print("                  │  (nBtags >= 1)  │  (nBtags >= 1)  │")
            print("                  ├─────────────────┼─────────────────┤")
            print("    mT > 50       │   Region C      │   Region D      │")
            print("                  │  (nBtags >= 1)  │  (nBtags >= 1)  │")
            print("                  └─────────────────┴─────────────────┘")
        else:
            print(f"Method               : CutBasedID vs nBtags")
            print(f"mT requirement       : > 50 for all regions")
            print(f"================FLOWCHART=========================")
            print("                      ID =Loose       ID = Tight")
            print("                  ┌─────────────────┬─────────────────┐")
            print("    nBtags = 0    │   Region A      │   Region B      │")
            print("                  │  (mT > 50)      │  (mT > 50)      │")
            print("                  ├─────────────────┼─────────────────┤")
            print("    nBtags >=1    │   Region C      │   Region D      │")
            print("                  │  (mT > 50)      │  (mT > 50)      │")
            print("                  └─────────────────┴─────────────────┘")

        
        print(f"Region               : {self.region}")
        print(f"Year                 : {self.year}")
        print(f"Jet pT threshold     : {self.JetPt} GeV")
        print(f"B-tag WP             : {self.bTagWP} ({self.DeepJetWP:.4f})")
        print(f"CutBased ID          : {self.IdWP} ({self.CutBasedIdTight})")
        print(f"mT cut               : {self.mT_operator} {self.mT} GeV")
        print(f"nBtags cut           : {self.nbtags_operator} {self.nbtags_threshold}")
        print("="*60 + "\n")

        #------Check for events.HLT.fields for correct trigger -------#
        def get_trigger_mask(hlt ,era):
            if era in ["2016postVFP","2016preVFP"]:
                return hlt.Ele32_eta2p1_WPTight_Gsf

            elif era in ["2018"]:
                return hlt.Ele32_WPTight_Gsf

            elif era in ["2017"]:
                return hlt.Ele35_WPTight_Gsf

            else:
                raise ValueError(f"The era you are trying to analyse is not present ! please pick among {self.available_eras}")

        #------Check the official twiki page for Pileup jet Ids for RUN2 UL------#
        def get_puId_mask(jets ,era):
            if era in self.available_eras:
                puId_mask = (jets.puId & self.PileUpConfig[era][self.PileUpWP])!=0
                return puId_mask

            else : 
                raise ValueError(f"The era you are trying to analyse is not present ! please pick among {self.available_eras}")
        
        #------For Good Jets ---------- #
        def get_good_jets(jets):
            has_tight_Id = (jets.jetId & (1 << 2))!= 0
            
            #----apply puId only for jets with pt < 50 ------#
            has_puId = get_puId_mask(jets,self.year)
            passes_puId = dak.where(jets.pt > 50 , True , has_puId)
            
            is_jet = (jets.pt >= self.JetPt) & (abs(jets.eta) < self.JetEta) & has_tight_Id & passes_puId 
            
            return is_jet
        
        #------For Good Electrons ------#
        def get_good_electrons(electrons):
            basic_cuts = (
                (electrons.pt>=self.ElectronPt)& (abs(electrons.eta) < self.ElectronEta) & ~((abs(electrons.eta) >= self.barrel) & (abs(electrons.eta) <= self.endcap)) & (electrons.cutBased == self.CutBasedIdTight))
            
            is_barrel = abs(electrons.eta) < self.barrel
            
            max_dxy = dak.where(is_barrel,0.05,0.10)
            max_dz = dak.where(is_barrel,0.10,0.20)
            
            ip_cuts = (abs(electrons.dxy) <= max_dxy) & (abs(electrons.dz) <= max_dz)
            
            return basic_cuts & ip_cuts
        
        # ------- Select Good Objects --------------------------------#
        good_electrons = events.Electron[get_good_electrons(events.Electron)]
        leading_electron = dak.firsts(good_electrons)
        met = events.MET
        
        # ---------Calculate Transverse Mass -------------------------#
        dphi = leading_electron.phi - met.phi
        dphi = (dphi + np.pi)%(2*np.pi) - np.pi
        mT = np.sqrt(2 * leading_electron.pt * met.pt * (1 - np.cos(dphi)))
        mT_mask = self.ops[self.mT_operator](mT, self.mT)
        mT_mask_filtered = dak.fill_none(mT_mask , False)

        # ------- B-Tagging Selection --------------------------------#
        is_btagged = (events.Jet.btagDeepFlavB > self.DeepJetWP) & get_good_jets(events.Jet)
        nbtags_num = dak.sum(is_btagged, axis=1)
        nbtags_mask = self.ops[self.nbtags_operator](nbtags_num, self.nbtags_threshold)

        # ------- Build Selection ------------------------------------#
        selection.add_multiple({
            "Trigger": get_trigger_mask(events.HLT,self.year),
            "good_electrons": dak.sum(get_good_electrons(events.Electron),axis=1) > self.nElectron , 
            "has_atleast_4jets":dak.sum(get_good_jets(events.Jet), axis=1) >= self.nJets,
            "nbtags_criteria":nbtags_mask,
            "mT_criteria":mT_mask_filtered
        })
        
        # --------Apply Full Selection---------------------------------#
        mask = selection.all("Trigger", "good_electrons", "has_atleast_4jets","nbtags_criteria","mT_criteria")
        
        # --------Select Objects After All Cuts -----------------------#
        selected_jets = events.Jet[mask]
        sjets = selected_jets[get_good_jets(selected_jets)]
        
        ele = events.Electron[mask]
        sel = dak.firsts(ele[get_good_electrons(ele)])
        
        #-------sorting sjets based on their pt-------------# 
        sjet_sorted = dak.argsort(sjets.pt , ascending=False)
        sjets = sjets[sjet_sorted]
       
        gen = events.GenPart[mask]
        
        #------- Feature extraction------------------#
        
        pt = sjets.pt
        eta = sjets.eta
        phi = sjets.phi

        cosh_eta = np.cosh(eta)
        sinh_eta = np.sinh(eta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        p_mag = pt * cosh_eta
        
        #----------accumulators--------------------#
        jetHT = dak.sum(pt, axis=1)
        sqrt_s = dak.sum(p_mag, axis=1)
        sum_s = dak.sum(p_mag*p_mag, axis=1)
    
        #----------Rectangular Components----------#
        px = pt * cos_phi
        py = pt * sin_phi 
        pz = pt * sinh_eta
    
        #----------DeltaR Calculation--------------#
        px_total = dak.sum(px, axis=1)
        py_total = dak.sum(py, axis=1)
        pz_total = dak.sum(pz, axis=1)

        phi_system = np.arctan2(py_total, px_total)
        pt_system = np.sqrt(px_total**2 + py_total**2)

        eta_system = np.arcsinh(pz_total / pt_system)
        
        delta_eta = sel.eta - eta_system
        delta_phi_raw = sel.phi - phi_system
    
        delta_phi = (delta_phi_raw + np.pi)%(2*np.pi) - np.pi
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        #----------Compute AL----------------------#
        AL_num = dak.sum(pt*sinh_eta, axis=1)
        AL = dak.where(sqrt_s > 0, AL_num / sqrt_s, 0.0)

        #----------Sphericity Calculation-----------#
        SphericityCaps = namedtuple('SphericityCaps', ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'])
        def compute_sphericity_values(c1,c2,c3,d):
            def Sphericity(mom1 ,mom2):
                return dak.sum((mom1 * mom2) , axis=1)

            components = [c1,c2,c3]

            #compute raw values
            values = [
                Sphericity(components[i],components[j])
                for i in range(3) for j in range(i,3)
            ]

            #compute normalised values
            normalised_values = [
                dak.where(d>0, val/d ,0.0)
                for val in values
            ]

            return SphericityCaps(*normalised_values)


        S = compute_sphericity_values(px,py,pz,sum_s)
        Sxx ,Sxy, Sxz, Syy ,Syz ,Szz = S
        
        #-------------Fox Wolfram Moments Calculation--------#
        def combinations(var):
            return var[:, :, None] * var[:, None, :]

        dot_prod = (combinations(px) + combinations(py) + combinations(pz))
        den = combinations(p_mag)
        
        cos_theta = dak.where(den>0, dot_prod/den ,0.0)
        
        w = den
        P_l = {
            0 : 1,
            1 : cos_theta,
            2 : 0.5 * (3 * cos_theta**2 -1),
            3 : 0.5 * (5 * cos_theta**3 - 3*cos_theta)
        }

        def Hl_pair(l):
            return w * P_l[l]
        
        def Hl(Hl_pair):
            return dak.sum(dak.sum(Hl_pair,axis=2),axis=1)

        # Evaluate Hl_pair
        H1_pair = Hl_pair(1)
        H2_pair = Hl_pair(2)
        H3_pair = Hl_pair(3)

        # Evaluate HL
        H1 = Hl(H1_pair)
        H2 = Hl(H2_pair)
        H3 = Hl(H3_pair)
        
        # Evaluate FW_l
        def FW_l(Hl):
            return dak.where(sqrt_s > 0 , Hl/(sqrt_s*sqrt_s),0.0)

        FW1 = FW_l(H1)
        FW2 = FW_l(H2)
        FW3 = FW_l(H3)

        # Total transverse activity
        met = events.MET[mask]
        met_pt = met.pt
        pTSum = jetHT + met_pt
        
        print("3..")
        #Construct Sphericity Tensor 
        def build_rows(S1,S2,S3):
            return dak.concatenate([S1[:,None], S2[:,None],S3[:,None]],axis=1)
        
        row1 = build_rows(Sxx,Sxy,Sxz)
        row2 = build_rows(Sxy,Syy,Syz)
        row3 = build_rows(Sxz,Syz,Szz)


        #Stack Rows 
        Sphericity_tensor = dak.concatenate([
            row1[:,None,:],
            row2[:,None,:],
            row3[:,None,:]
        ],axis=1
        )

        def wrap_eigvals(array):
            out = np.linalg.eigvals(ak.typetracer.length_zero_if_typetracer(array))
            if ak.backend(array) == "typetracer":
                out = ak.Array(out.layout.to_typetracer(forget_length=True))
            
            return out

        eigenvalues = dak.map_partitions(wrap_eigvals,Sphericity_tensor)

        sorted_eval = eigenvalues[dak.argsort(eigenvalues, ascending=False)]

        lambda1 = sorted_eval[:,0]
        lambda2 = sorted_eval[:,1]
        lambda3 = sorted_eval[:,2]

        def lambda_related(num,den):
            return ak.where(den > 0 ,num/den,0.0)
            
        sphericity_val = 1.5 * (lambda2 + lambda3)
        planarity = lambda_related(lambda3,lambda2)
        alignment = lambda_related(lambda2,lambda1)
        
        nJet = dak.num(sjets.pt, axis=1)
        p2in = lambda_related(lambda2,nJet)
        p2out = lambda_related(lambda3,nJet)

        pdg_id_array = events.GenPart[mask].pdgId
        first_two = pdg_id_array[:,:2]
        sum_pdgId=dak.sum(first_two, axis=1)
        

        #print("The array is  : ",pdg_id_array.compute())
        bjets = sjets[sjets.btagDeepFlavB > 0.2489]
        bjets_sorted = bjets[dak.argsort(bjets.pt, ascending=False)]
        leading_bjet = dak.firsts(bjets_sorted)

        dphi_lb_raw = sel.phi - leading_bjet.phi
        dphi_lb = (dphi_lb_raw + np.pi)%(2*np.pi) - np.pi
        
        # bdt container 
        bdt_inputs = {
            "FW1":FW1,
            "Sxz":Sxz,
            "Szz":Szz,
            "AL":AL,
            "p2in":p2in,
            "planarity":planarity,
            "pT_Sum":pTSum,
            "nJet":nJet,
            "delta_R":delta_R,
            "dphi_lb":dphi_lb,
        }

        bdt_inputs = {k: ak.fill_none(v, 0.0) if k in ("delta_R","dphi_lb") else v for k, v in bdt_inputs.items()}
        
        bdt_events = ak.zip(bdt_inputs, depth_limit=1)
        bdt_score = self.bdt(bdt_events)
        #print(bdt_score.compute())
        # Beginning of reconstruction 
        recoselection = PackedSelection()
        
        recoselection.add_multiple({
            "BDT_cut" : bdt_score > self.BDT_cut_val,
            "2btagged_jets": dak.sum(sjets.btagDeepFlavB > self.DeepJetWP, axis=1) >=2

        })

        reco_mask = recoselection.all("BDT_cut", "2btagged_jets")

        
        before_reco = events.Electron[mask]
        cleaning_reco = before_reco[get_good_electrons(before_reco)]
        reco_electrons = cleaning_reco[reco_mask]

        reco_jets = sjets[reco_mask]

        reco_gen = gen[reco_mask]
        #------------ choose the leading electron --------------------#
        leading_reco_el = dak.firsts(reco_electrons)

        reco_met = met[reco_mask]

        #------------choose 2 btagged jets first and the remaining jets as light jets------#
        reco_jets = reco_jets[dak.argsort(reco_jets.pt, ascending=False)]

        is_b = reco_jets.btagDeepFlavB > self.DeepJetWP
        reco_bjets = reco_jets[is_b]
        leading_two_bjets = reco_bjets[:,:2]

        idx = dak.local_index(reco_jets.pt, axis=1)
        bjets_index = idx[is_b]
        leading_bjets_idx = bjets_index[:,:2]

        excluding_mask = dak.any(idx[:,:,None] == leading_bjets_idx[:,None,:], axis=2)
        remaining_jets = reco_jets[~excluding_mask]

        leading_remaining_jets = remaining_jets[:,:2]

        b1 = leading_two_bjets[:,0]
        b2 = leading_two_bjets[:,1]
        l1 = leading_remaining_jets[:,0]
        l2 = leading_remaining_jets[:,1]

        # for weight calculation
        fields = [            
            "area", "btagCSVV2", "btagDeepB", "btagDeepFlavB", "chEmEF", "chHEF", "eta", "mass",
            "muEF", "muonSubtrFactor", "neEmEF", "neHEF", "phi", "pt", "qgl", "rawFactor",
            "bRegCorr", "bRegRes", "jetId", "nElectrons", "nMuons", "puId", "nConstituents",
            "genJetIdx", "hadronFlavour", "partonFlavour", "genJetIdxG"
        ]

        merged_jets = dak.zip({
            field : dak.concatenate([leading_two_bjets[field], leading_remaining_jets[field]], axis=1)
            for field in fields
        })

        #---------------Calculate the four momenta---------------------------------#
        def four_momenta(obj):
            return obj.px, obj.py, obj.pz, obj.energy

        # for electron and Jets 
        e_px, e_py, e_pz, e_E = four_momenta(leading_reco_el)
        b1_px, b1_py, b1_pz, b1_E = four_momenta(b1)
        b2_px, b2_py, b2_pz, b2_E = four_momenta(b2)
        l1_px, l1_py, l1_pz, l1_E = four_momenta(l1)
        l2_px, l2_py, l2_pz, l2_E = four_momenta(l2)

        nu_px = reco_met.px
        nu_py = reco_met.py
       #------------------ Compute the pz of neutrino------------------------------#
        a = self.MW**2 + 2 * (e_px*nu_px + e_py*nu_py)
        A = 4*(e_E**2-e_pz**2)
        B = -4*a*e_pz
        C = 4*e_E**2*(nu_px**2+nu_py**2)-a**2

        A = dak.where(abs(A) < 1e-6, A*0 + 1e-6, A)

        discr = B**2 - 4*A*C

        sqrt_discr = np.sqrt(np.maximum(discr, 0))

        pz_sol1 = -(B + sqrt_discr) / (2*A)
        pz_sol2 = -(B - sqrt_discr) / (2*A)

        nu_E_sol1 = np.sqrt(nu_px**2 + nu_py**2 + pz_sol1**2)
        nu_E_sol2 = np.sqrt(nu_px**2 + nu_py**2 + pz_sol2**2)

        # ---------------- Construct W had from the two light jets----------------#
        W_hadronic = l1 + l2
        W_hadronic_px, W_hadronic_py, W_hadronic_pz, W_hadronic_E = four_momenta(W_hadronic)

        bjets_4_mom = ak.zip({
            "px": dak.concatenate([b1_px[:,None],b2_px[:,None]],axis=1),
            "py": dak.concatenate([b1_py[:,None],b2_py[:,None]],axis=1),
            "pz": dak.concatenate([b1_pz[:,None],b2_pz[:,None]],axis=1),
            "E": dak.concatenate([b1_E[:,None],b2_E[:,None]],axis=1)
        })

        nu_pz_sols = dak.concatenate([pz_sol1[:,None],pz_sol2[:,None]],axis=1)
        nu_pz_E = dak.concatenate([nu_E_sol1[:,None],nu_E_sol2[:,None]],axis=1)
        
        # charge of the leading electron 
        electron_charge = leading_reco_el.charge

        # Gen level top and antitop (Only in the case of ttbar semileptonic)
        if dataset == "ttbar_SemiLeptonic":
            is_top = reco_gen[(reco_gen.pdgId == 6) & (reco_gen.hasFlags(["isLastCopy"]))]
            is_antitop = reco_gen[(reco_gen.pdgId == -6) & (reco_gen.hasFlags(["isLastCopy"]))]

            firsts_top = dak.firsts(is_top)
            gen_top_px = firsts_top.px
            gen_top_py = firsts_top.py
            gen_top_pz = firsts_top.pz
            gen_top_E = firsts_top.energy
            
            def calculate_pt_rapidity(px,py,pz,E):
                pt = np.sqrt(px**2 + py**2)
                y = 0.5 * np.log((E+pz)/(E-pz))
                
                return pt, y
        
            firsts_antitop = dak.firsts(is_antitop)
            gen_antitop_px = firsts_antitop.px
            gen_antitop_py = firsts_antitop.py
            gen_antitop_pz = firsts_antitop.pz
            gen_antitop_E = firsts_antitop.energy
            
            gen_top_pt, gen_top_rap = calculate_pt_rapidity(gen_top_px, gen_top_py, gen_top_pz, gen_top_E)
            gen_antitop_pt, gen_antitop_rap = calculate_pt_rapidity(gen_antitop_px, gen_antitop_py, gen_antitop_pz, gen_antitop_E)
            
            gen_top_mass = np.sqrt(gen_top_E**2 - (gen_top_px**2 + gen_top_py**2 + gen_top_pz**2))
            gen_antitop_mass = np.sqrt(gen_antitop_E**2 - (gen_antitop_px**2 + gen_antitop_py**2 + gen_antitop_pz**2))
            
            gen_top_cos_phi = np.cos(firsts_top.phi)
            gen_top_sin_phi = np.sin(firsts_top.phi)
            gen_top_sinh_eta = np.sinh(firsts_top.eta)
            
            gen_antitop_cos_phi = np.cos(firsts_antitop.phi)
            gen_antitop_sin_phi = np.sin(firsts_antitop.phi)
            gen_antitop_sinh_eta = np.sinh(firsts_antitop.eta)

        PERMUTATIONS = [(0,1,0),(0,1,1),(1,0,0),(1,0,1)]
        
        # chi2 recipes
        x_m_list = []
        res_list = []
        perm_idx = []
        
        #mass acccumulators 
        m_e_list = []
        m_nu_list = [] 
        m_b_lep_list = []
        m_b_had_list = []
        m_l1_list = []
        m_l2_list = []
        
        # charge containers 
        el_charge =[]

        # Gen containers
        if dataset == "ttbar_SemiLeptonic":
            gen_top_mass_list = []
            gen_antitop_mass_list = []
            
            gen_top_pt_list = []
            gen_antitop_pt_list = []
            
            gen_top_rap_list = []
            gen_antitop_rap_list = []
            
            gen_top_cos_phi_list = []
            gen_top_sin_phi_list = []
            gen_top_sinh_eta_list = []
            
            gen_antitop_cos_phi_list =[]
            gen_antitop_sin_phi_list = []
            gen_antitop_sinh_eta_list = []
            
            gen_top_E_list = []
            gen_antitop_E_list = []
        
        def resolutions(px,py,pz,n):
            res_px = n * abs(px)
            res_py = n * abs(py)
            res_pz = n * abs(pz)

            return res_px, res_py, res_pz

        for perm_id, (lep_b_idx, had_b_idx, nu_idx) in enumerate(PERMUTATIONS):
            # Select the bjets for this permutation  
            b_lep_px = bjets_4_mom.px[:,lep_b_idx]
            b_lep_py = bjets_4_mom.py[:,lep_b_idx]
            b_lep_pz = bjets_4_mom.pz[:,lep_b_idx]
            b_lep_E = bjets_4_mom.E[:,lep_b_idx]

            b_had_px = bjets_4_mom.px[:,had_b_idx]
            b_had_py = bjets_4_mom.py[:,had_b_idx]
            b_had_pz = bjets_4_mom.pz[:,had_b_idx]
            b_had_E = bjets_4_mom.E[:,had_b_idx]

            # select the neutrino pz solution for this permutation  
            nu_pz = nu_pz_sols[:,nu_idx]
            nu_E = nu_pz_E[:,nu_idx]

            # Build the x_m = [px,py,pz * 18]
            x_m = dak.concatenate([e_px[:,None],e_py[:,None],e_pz[:,None],nu_px[:,None],nu_py[:,None],nu_pz[:,None],b_lep_px[:,None],b_lep_py[:,None],b_lep_pz[:,None],b_had_px[:,None],b_had_py[:,None],b_had_pz[:,None],l1_px[:,None],l1_py[:,None],l1_pz[:,None],l2_px[:,None],l2_py[:,None],l2_pz[:,None]],axis=1)

            res_e_px, res_e_py, res_e_pz = resolutions(e_px,e_py,e_pz,self.res_values["electron"])
            res_nu_px, res_nu_py, res_nu_pz = resolutions(nu_px, nu_py, nu_pz,self.res_values["MET"])
            res_blep_px, res_blep_py, res_blep_pz = resolutions(b_lep_px, b_lep_py, b_lep_pz,self.res_values["Jet"])
            res_bhad_px, res_bhad_py, res_bhad_pz = resolutions(b_had_px, b_had_py, b_had_pz,self.res_values["Jet"])
            res_l1_px, res_l1_py, res_l1_pz = resolutions(l1_px, l1_py, l1_pz,self.res_values["Jet"])
            res_l2_px, res_l2_py, res_l2_pz = resolutions(l2_px, l2_py, l2_pz,self.res_values["Jet"])

            res = dak.concatenate([res_e_px[:,None], res_e_py[:,None], res_e_pz[:,None],
                                   res_nu_px[:,None], res_nu_py[:,None], res_nu_pz[:,None],
                                   res_blep_px[:,None], res_blep_py[:,None], res_blep_pz[:,None],
                                   res_bhad_px[:,None], res_bhad_py[:,None], res_bhad_pz[:,None],
                                   res_l1_px[:,None], res_l1_py[:,None], res_l1_pz[:,None],
                                   res_l2_px[:,None], res_l2_py[:,None], res_l2_pz[:,None]], axis=1)
           

            # Store the masses as well 
            #m_e = np.zeros_like(e_px)
            b_lep_mass = np.sqrt(b_lep_E**2 - (b_lep_px**2 + b_lep_py**2 + b_lep_pz**2))
            b_had_mass = np.sqrt(b_had_E**2 - (b_had_px**2 + b_had_py**2 + b_had_pz**2))
            l1_mass = np.sqrt(l1_E**2 - (l1_px**2 + l1_py**2 + l1_pz**2))
            l2_mass = np.sqrt(l2_E**2 - (l2_px**2 + l2_py**2 + l2_pz**2))
                
            x_m_list.append(x_m)
            res_list.append(res)
            perm_idx.append(perm_id)
            el_charge.append(electron_charge)

            # store the masses  
            #m_e_list.append(m_e)
            m_b_lep_list.append(b_lep_mass)
            m_b_had_list.append(b_had_mass)
            m_l1_list.append(l1_mass)
            m_l2_list.append(l2_mass)

            # store the gen level top and anti-top mass, pt, eta, phi(Only in case of ttbar_SemiLeptonic)
            if dataset == "ttbar_SemiLeptonic":
                gen_top_mass_list.append(gen_top_mass)
                gen_antitop_mass_list.append(gen_antitop_mass)
                
                gen_top_pt_list.append(gen_top_pt)
                gen_antitop_pt_list.append(gen_antitop_pt)
                
                gen_top_rap_list.append(gen_top_rap)
                gen_antitop_rap_list.append(gen_antitop_rap)
                
                gen_top_cos_phi_list.append(gen_top_cos_phi)
                gen_top_sin_phi_list.append(gen_top_sin_phi)
                gen_top_sinh_eta_list.append(gen_top_sinh_eta)
                
                gen_antitop_cos_phi_list.append(gen_antitop_cos_phi)
                gen_antitop_sin_phi_list.append(gen_antitop_sin_phi)
                gen_antitop_sinh_eta_list.append(gen_antitop_sinh_eta)
                
                gen_top_E_list.append(gen_top_E)
                gen_antitop_E_list.append(gen_antitop_E)



        # Weight files
        weights = Weights(size= None, storeIndividual=True)

        events_reco = events[mask][reco_mask]
        
        if isRealData:
            print(f"RealData block is being executed for {dataset}.....")
            if hasattr(events, "L1PreFiringWeight"):
                print("PrefiringWeight is present")
                weights.add("PreFiringWeight",weight=events_reco.L1PreFiringWeight.Nom, weightUp=events_reco.L1PreFiringWeight.Up, weightDown=events_reco.L1PreFiringWeight.Dn)

            else:
                print("Prefiring weight is not present")
                weights.add("PreFiringWeight", weight=dak.ones_like(leading_reco_el.pt))

            print(list(weights._weights.keys()))

        else:
            print(f"MC block is being executed for {dataset}.....")
            if hasattr(events, "puWeight"):
                print("pileup weight is present")
                weights.add("pileupWeight",weight=events_reco.puWeight, weightUp=events_reco.puWeightUp, weightDown=events_reco.puWeightDown)
            else:
                print("pileup weight is not present")
                weights.add("pileupWeight", weight=dak.ones_like(leading_reco_el.pt))
                
            if hasattr(events, "L1PreFiringWeight"):
                print("L1prefiring weight is present")
                weights.add("PreFiringWeight",weight=events_reco.L1PreFiringWeight.Nom, weightUp=events_reco.L1PreFiringWeight.Up, weightDown=events_reco.L1PreFiringWeight.Dn)
            else:
                print("Prefiring weight is not present")
                weights.add("PreFiringWeight", weight=dak.ones_like(leading_reco_el.pt))
                    
            if hasattr(events, "LHEWeight"):
                print("LHEWeight is present")
                weights.add("LHEWeight",weight=events_reco.LHEWeight.originalXWGTUP / abs(events_reco.LHEWeight.originalXWGTUP))
            else:
                print("LHEWeight is not present")
                weights.add("LHEWeight", weight=dak.ones_like(leading_reco_el.pt))

            #---------location of the root and the json files------#                                                                                                                                               
            Dir = "/nfs/home/sanskar/SF/SFs"
            Dir1 = f"/nfs/home/sanskar/SF/SFs/{self.year}/WITH_NBTAGS_VS_ID"
            filename1 = f"UL{self.year}_el_HLT_{self.HLT_scalefactor}.root"
            filename2 = f"UL{self.year}_el_ID.json"
            filename3 = f"UL{self.year}_jet_Btagging.json"
            filename4 = f"btag_eff_region{self.region}_{self.year}_correctionlib_with_id_and_nbtags.json"
            filename5 = f"UL{self.year}_jet_jmar.json"

            filename3a = "UL2016preVFP_jet_Btagging.json"
            #----------Concatenate File Paths----------------------#                                                                                                                                               
            full_path1 = os.path.join(Dir, filename1)
            full_path2 = os.path.join(Dir, filename2)
            full_path3 = os.path.join(Dir ,filename3)
            full_path4 = os.path.join(Dir1 ,filename4)
            full_path5 = os.path.join(Dir ,filename5)

            full_path3a = os.path.join(Dir,filename3a)

            #------------HLT scale factors ------------------------#
            if os.path.isfile(full_path1):
                ext = extractor()
                ext.add_weight_sets([f"* * {full_path1}"])
                ext.finalize()
                evaluator = ext.make_evaluator()
                electron_sf = evaluator["EGamma_SF2D"](leading_reco_el.eta, leading_reco_el.pt)
                weights.add("HLT_SF", weight=electron_sf)
                print("HLT scale factors :applied ")
            else:
                print("HLT scale_factors: not present adding +1 to weight")
                weights.add("HLT_SF", weight=dak.ones_like(leading_reco_el.pt))

            # ------------ electron id sf -------------------------#
            if os.path.isfile(full_path2):
                eset = CorrectionSet.from_file(full_path2)
                ele_corr = eset["UL-Electron-ID-SF"]
                
                reco_sf = ele_corr.evaluate(self.year, "sf", "RecoAbove20", leading_reco_el.eta, leading_reco_el.pt)
                reco_sf_up = ele_corr.evaluate(self.year, "sfup", "RecoAbove20", leading_reco_el.eta, leading_reco_el.pt)
                reco_sf_down = ele_corr.evaluate(self.year, "sfdown", "RecoAbove20", leading_reco_el.eta, leading_reco_el.pt)
                
                id_sf = ele_corr.evaluate(self.year, "sf", self.IdWP, leading_reco_el.eta, leading_reco_el.pt)
                id_sf_up = ele_corr.evaluate(self.year, "sfup", self.IdWP, leading_reco_el.eta, leading_reco_el.pt)
                id_sf_down = ele_corr.evaluate(self.year, "sfdown", self.IdWP, leading_reco_el.eta, leading_reco_el.pt)
                
                ele_id_sf = reco_sf * id_sf
                ele_id_sf_up = reco_sf_up * id_sf_up
                ele_id_sf_down = reco_sf_down * id_sf_down

                weights.add("eleID", weight=ele_id_sf, weightUp=ele_id_sf_up, weightDown=ele_id_sf_down)
                print("ele_id : applied succesfully")

            else :
                print("ele_id :not present , adding +1 to weights")
                weights.add("eleID" , weight = dak.ones_like(leading_reco_el.pt))


            # ------------------ btag scale factors --------------------#
            if os.path.isfile(full_path3):
                # load the b-tagging correction                                                                                                                                                                   
                cset= CorrectionSet.from_file(full_path3)
                btag_corr_heavy = cset["deepJet_comb"]
                
                if self.year == "2016postVFP":
                    cset_preVFP = CorrectionSet.from_file(full_path3a)
                    btag_corr_light = cset_preVFP["deepJet_incl"]
                else:
                    btag_corr_light = cset["deepJet_incl"]
                    
                #---Load the b-tagging efficiencies----                                                                                                                                                       
                bset = CorrectionSet.from_file(full_path4)
                btag_eff = bset[f"btag_eff_region{self.region}_DeepJet_Medium"]
                
                is_light = merged_jets.hadronFlavour == self.lflav
                is_heavy = (merged_jets.hadronFlavour == self.bflav)|(merged_jets.hadronFlavour == self.cflav)
                    
                flavor_for_heavy = dak.where(is_heavy , merged_jets.hadronFlavour ,self.bflav)
                flavor_for_light = dak.where(is_light , merged_jets.hadronFlavour ,self.lflav)
                
                jet_eff = btag_eff.evaluate(dataset , merged_jets.hadronFlavour , abs(merged_jets.eta) ,merged_jets.pt)
                
                tagged = merged_jets.btagDeepFlavB > self.DeepJetWP
                
                p_mc = dak.where(tagged , jet_eff , 1-jet_eff)
                p_mc_event = dak.prod(p_mc ,axis=1)
                    
                event_weight = {}
                for variation in ["central","up","down"]:
                    sf_heavy = btag_corr_heavy.evaluate(variation, self.bTagWP, flavor_for_heavy, abs(merged_jets.eta), merged_jets.pt)
                    sf_light = btag_corr_light.evaluate(variation, self.bTagWP, flavor_for_light, abs(merged_jets.eta), merged_jets.pt)
                    
                    jet_sf = dak.where(is_heavy, sf_heavy ,sf_light)
                    
                    eff_data = jet_eff * jet_sf
                    
                    p_data = dak.where(tagged, eff_data ,1-eff_data)
                    p_data_event = dak.prod(p_data ,axis=1)
                    
                    event_weight[variation] = dak.where(p_mc_event >0 , p_data_event/p_mc_event ,1.0)
                    
                weights.add("btag_sf",weight = event_weight["central"], weightUp = event_weight["up"] ,weightDown = event_weight["down"])
                print("Btagging Corrections : APPLIED")
            else:
                print("Btagging Corrections : NOT APPLIED")
                weights.add("btag_sf", weight = dak.ones_like(leading_reco_el.pt))
            
            # --------------- Pileup Jet id ------------------------------------------#
            if os.path.isfile(full_path5):
                pset = CorrectionSet.from_file(full_path5)
                pjetid_corr= pset["PUJetID_eff"]

                puId_jet_pt_mask = merged_jets.pt <= 50

                dummy_pt = dak.where(merged_jets.pt > 50, 50, merged_jets.pt)
                puId_weight = {}

                for variation in ["nom","up","down"]:
                    sf_puId = pjetid_corr.evaluate(merged_jets.eta ,dummy_pt ,variation , self.PileUpWP)
                    sf = dak.where(puId_jet_pt_mask , sf_puId ,1.0)
                    
                    sf_multiplied = dak.prod(sf, axis=1)
                    
                    puId_weight[variation] = sf_multiplied

                weights.add("pileUp_sf",weight = puId_weight["nom"], weightUp=puId_weight["up"],weightDown=puId_weight["down"])
                print("PILE-UP JETID CORRECTIONS: APPLIED")
            else:
                print("PILEUP JET-ID CORRECTIONS: NOT APPLIED")
                weights.add("pileUp_sf",weight = dak.ones_like(leading_reco_el.pt))


            print(list(weights._weights.keys()))

            
        x_m_all = dak.concatenate(
            [x[:, None, :] for x in x_m_list],
            axis=1
        )

        res_all = dak.concatenate(
            [r[:, None, :] for r in res_list],
            axis=1
        )


        perm_idx_all = dak.concatenate(
            [dak.full_like(res_list[0][:, None, 0], i) for i in range(4)],
             axis=1
        )

        
        e_charge_all = dak.concatenate(
            [e[:,None] for e in el_charge],
            axis=1
        )
        

        m_b_lep_all = dak.concatenate(
            [m[:,None] for m in m_b_lep_list],
            axis=1

        )

        m_b_had_all = dak.concatenate(
            [m[:,None] for m in m_b_had_list],
            axis=1
        )

        m_l1_all = dak.concatenate(
            [m[:,None] for m in m_l1_list],
            axis=1
        )

        m_l2_all = dak.concatenate(
            [m[:,None] for m in m_l2_list],
            axis=1
        )

        if dataset == "ttbar_SemiLeptonic":
            m_gtop_all = dak.concatenate(
                [m[:,None] for m in gen_top_mass_list],
                axis=1
            )
            
            m_gantitop_all = dak.concatenate(
                [m[:,None] for m in gen_antitop_mass_list],
                axis=1
            )
            
            gtop_pt_all = dak.concatenate(
                [pt[:,None] for pt in gen_top_pt_list],
                axis=1
            )
            
            gantitop_pt_all = dak.concatenate(
                [pt[:,None] for pt in gen_antitop_pt_list],
                axis=1
            )
            
            gtop_rap_all = dak.concatenate(
                [rap[:,None] for rap in gen_top_rap_list],
                axis=1
            )
            
            gantitop_rap_all = dak.concatenate(
                [rap[:,None] for rap in gen_antitop_rap_list],
                axis=1
            )
            
            gtop_cosphi_all = dak.concatenate(
                [cosphi[:,None] for cosphi in gen_top_cos_phi_list],
                axis=1
            )
            
            gtop_sinphi_all = dak.concatenate(
                [sinphi[:,None] for sinphi in gen_top_sin_phi_list],
                axis=1
            )
            
            gtop_sinheta_all = dak.concatenate(
                [sinh_eta[:,None] for sinh_eta in gen_top_sinh_eta_list],
                axis=1
            )
            
            gtop_E_all = dak.concatenate(
                [E[:,None] for E in gen_top_E_list],
                axis=1
            )
            
            gantitop_cosphi_all = dak.concatenate(
                [cosphi[:,None] for cosphi in gen_antitop_cos_phi_list],
                axis=1
            )
            
            gantitop_sinphi_all = dak.concatenate(
                [sinphi[:,None] for sinphi in gen_antitop_sin_phi_list],
                axis=1
            )
            
            gantitop_sinheta_all = dak.concatenate(
                [sinheta[:,None] for sinheta in gen_antitop_sinh_eta_list],
                axis=1
            )
            
            gantitop_E_all = dak.concatenate(
                [E[:,None] for E in gen_antitop_E_list],
                axis=1
            )

        weights_all = weights.weight()
        
        #----------------Return Computed Results----------------------#
        common_dict = {
            "x_m": x_m_all,
            "res": res_all,
            "perm_idx": perm_idx_all,
            "b_lep_mass": m_b_lep_all,
            "b_had_mass": m_b_had_all,
            "l1_mass": m_l1_all,
            "l2_mass": m_l2_all,
            "charge": e_charge_all,
            "weights": weights_all,
        }

        if dataset == "ttbar_SemiLeptonic":
            common_dict.update({
                "gen_top_mass": m_gtop_all,
                "gen_antitop_mass": m_gantitop_all,
                "gen_top_pt": gtop_pt_all,
                "gen_antitop_pt": gantitop_pt_all,
                "gen_top_rap": gtop_rap_all,
                "gen_antitop_rap": gantitop_rap_all,
                "gen_top_cosphi": gtop_cosphi_all,
                "gen_top_sinphi": gtop_sinphi_all,
                "gen_top_sinheta": gtop_sinheta_all,
                "gen_top_E": gtop_E_all,
                "gen_antitop_cosphi": gantitop_cosphi_all,
                "gen_antitop_sinphi": gantitop_sinphi_all,
                "gen_antitop_sinheta": gantitop_sinheta_all,
                "gen_antitop_E": gantitop_E_all,
            })
        return {dataset: common_dict}

        #return{dataset:{"x_m":x_m_all,"res":res_all,"perm_idx":perm_idx_all,"b_lep_mass":m_b_lep_all,"b_had_mass":m_b_had_all,"l1_mass":m_l1_all,"l2_mass":m_l2_all,"charge":e_charge_all,"gen_top_mass":m_gtop_all,"gen_antitop_mass":m_gantitop_all, "gen_top_pt":gtop_pt_all, "gen_antitop_pt": gantitop_pt_all, "gen_top_rap":gtop_rap_all, "gen_antitop_rap":gantitop_rap_all,"gen_top_cosphi":gtop_cosphi_all, "gen_top_sinphi":gtop_sinphi_all,"gen_top_sinheta":gtop_sinheta_all, "gen_top_E":gtop_E_all, "gen_antitop_cosphi":gantitop_cosphi_all, "gen_antitop_sinphi":gantitop_sinphi_all, "gen_antitop_sinheta":gantitop_sinheta_all, "gen_antitop_E":gantitop_E_all, "weights":weights_all}}
        

    def postprocess(self, accumulator):
        return accumulator
