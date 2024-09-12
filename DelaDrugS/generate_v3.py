import torch
import preProcess
import torch.nn as nn
import random
import sys
import rdkit.Chem as Chem
import rdkit.Chem.rdchem
import SAScore
import os
import enchant
import rdkit.Chem.Crippen as cri
from rdkit.Chem import rdFingerprintGenerator

import selfies

selfies.bond_constraints.set_semantic_constraints({'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 3, 'B': 3, 'B+1': 2, 'B-1': 4, 'O': 2, 'O+1': 3, 'O-1': 1, 'N': 3, 'N+1': 4, 'N-1': 2, 'C': 4, 'C+1': 5, 'C-1': 3, 'P': 5, 'P+1': 6, 'P-1': 4, 'S': 6, 'S+1': 7, 'S-1': 5, '?': 8})

vocSelfie = {'A':'[C]', 'B':'[11C]', 'C':'[13CH1]', 'D':'[3H]', 'E':'[O]', 'F':'[C-1]', 'G':'[17F]', 'H':'[Branch2]', 'I':'[=P]', 'J':'[#11C]', 'K':'[11CH3]', 'L':'[=S+1]', 'M':'[11CH2]', 'N':'[131I]', 'O':'[=N]', 'P':'[N]', 'Q':'[=Branch2]', 'R':'[#Branch2]', 'S':'[Br]', 'T':'[#C-1]', 'U':'[#N]', 'V':'[123I]', 'W':'[#N+1]', 'X':'[=13CH1]', 'Y':'[F]', 'Z':'[#C]', 'a':'[C+1]', 'b':'[14CH1]', 'c':'[I+1]', 'd':'[=S]', 'e':'[=N+1]', 'f':'[14CH2]', 'g':'[#14C]', 'h':'[N-1]', 'i':'[#Branch1]', 'j':'[O-1]', 'k':'[125I]', 'l':'[15N]', 'm':'[I]', 'n':'[P]', 'o':'[18F]', 'p':'[=Ring1]', 'q':'[11CH1]', 'r':'[=C]', 's':'[P+1]', 't':'[75Br]', 'u':'[18OH1]', 'v':'[S+1]', 'w':'[Cl]', 'x':'[=N-1]', 'y':'[SH0]', 'z':'[=O+1]', '0':'[CH1-1]', '1':'[=SH0]', '2':'[=14CH1]', '3':'[=Ring2]', '4':'[35S]', '5':'[2H]', '6':'[14C]', '7':'[=18O]', '8':'[Branch1]', '9':'[14CH3]', '<':'[N+1]', '>':'[=Branch1]', '&':'[=O]', '!':'[O+1]', '=':'[32P]', '+':'[=14C]', '-':'[Ring2]', '[':'[13C]', ']':'[=11C]', '{':'[Ring1]', '}':'[S]', '/':'[NH1]', '_':'[124I]', '£':'[15NH1]'}

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)

class generatorWithSubstitutions():
    def __init__(self, paramsDict):
        self.device = paramsDict["device"]
        self.name = paramsDict["name"]
        self.network = torch.load(paramsDict["networkPath"], map_location=torch.device('cpu')).to(self.device)
        fh = open(paramsDict["vocabularyPath"])
        self.voc = eval(fh.read())
        fh.close()
        self.contextLength= paramsDict["contextLength"]
        self.vocLength = len(self.voc)
        self.softmax = nn.Softmax(dim = -1)
        fin = open(self.name,"r")
        molin = fin.read().split('\n')
        fin.close()
        molin_ok = []
        selfie = []
        for i in range(len(molin)):
            if (len(molin[i])>0):
                 try:
                    molll = Chem.MolFromSmiles(molin[i])
                    Chem.RemoveStereochemistry(molll)
                    molin_ok.append(Chem.MolToSmiles(molll))
                    se = selfies.encoder(Chem.MolToSmiles(molll))
                    for i in vocSelfie.values():
                        se = se.replace(i,list(vocSelfie.keys())[list(vocSelfie.values()).index(i)])
                    selfie.append(se)
                    #print (se,Chem.MolToSmiles(molll))
                 except:
                     continue

        self.molin_ok = molin_ok
        self.inputMole = selfie
        self.wantedValids = paramsDict["requestedValid"]
        self.pos = paramsDict["positions"]
        #Qesto parametro va tenuto d'occhio!
        self.N = int(self.wantedValids)*1000


    def run(self):
        thisPath = os.getcwd()
        outfff = ""
        for iii in range(len(self.molin_ok)):
            ref = fpgen.GetFingerprint(Chem.MolFromSmiles(self.molin_ok[iii]))
            outmol = []
            molFile = ""
            self.positions = random.sample(range(len(self.inputMole[iii])),min(len(self.inputMole[iii]),self.pos))
            query = preProcess.wordToTensor(self.inputMole[iii], self.voc)
            qs = query.shape[0]
            vocLen = len(self.voc)
            for z in range(0, qs):
                if preProcess.tensorToCategory(query[z], self.voc) == "~":
                    maxQ = z
                    break
            j = 0
            valids = 0
            diffFlag = 0
            while valids < int(self.wantedValids) and j<=self.N:
                outT = 0
                context = torch.zeros(1,self.contextLength*len(self.voc)).to(self.device)
                cellT1 = torch.zeros(1, self.network.hiddenSize_L1, device = self.device)
                cellT2 = torch.zeros(1, self.network.hiddenSize_L2, device = self.device)
                hiddenT1 = torch.zeros(1, self.network.hiddenSize_L1, device = self.device)
                hiddenT2 = torch.zeros(1, self.network.hiddenSize_L2, device = self.device)
                answer = "<s>"
                context = preProcess.addToContext(context, preProcess.categoryToTensor("<s>", self.voc).unsqueeze(0).to(self.device))
            #   #Determino le posizioni da sostituire
            # Not today!
                offset = 0
                for letter in range(0, qs):
                    if letter in self.positions:
                        #Aggiungere il tiro di dado
                        move = random.randint(1,3)
                        #print(move)

                        #caso 1: sostituzione
                        if (move==1):
                            charT, hiddenT1, cellT1, hiddenT2, cellT2 = self.network(context, hiddenT1, cellT1, hiddenT2, cellT2)
                            charTemp = self.softmax(charT)
                            s = 0
                            rn = random.uniform(0,1)
                            for vocPosition in range(0, vocLen):
                                maxv, maxp = torch.max(charTemp, dim = -1)
                                s = s + maxv.item()
                                if s >= rn:
                                    p = maxp.item()
                                    break
                                else:
                                    charTemp[0][maxp.item()] = 0

                            outT = torch.zeros(len(self.voc)).to(self.device)
                            outT[p] = 1
                        
                        
                            context = preProcess.addToContext(context, outT.unsqueeze(0))
                            answer = answer + preProcess.tensorToCategory(outT, self.voc)
                            if preProcess.tensorToCategory(outT, self.voc) in ["</s>","~"] or len(answer)>=200:
                                break
                        #caso 2: aggiunta    
                        if (move==2):
                        
                            answer = answer + preProcess.tensorToCategory(query[letter], self.voc)
                            charT, hiddenT1, cellT1, hiddenT2, cellT2 = self.network(context, hiddenT1, cellT1, hiddenT2, cellT2)
                            context = preProcess.addToContext(context, query[letter].unsqueeze(0))

                            if preProcess.tensorToCategory(query[letter], self.voc) in ["</s>","~"] or len(answer)>=200:
                                break

                            charT, hiddenT1, cellT1, hiddenT2, cellT2 = self.network(context, hiddenT1, cellT1, hiddenT2, cellT2)
                            charTemp = self.softmax(charT)
                            s = 0
                            rn = random.uniform(0,1)
                            for vocPosition in range(0, vocLen):
                                maxv, maxp = torch.max(charTemp, dim = -1)
                                s = s + maxv.item()
                                if s >= rn:
                                    p = maxp.item()
                                    break
                                else:
                                    charTemp[0][maxp.item()] = 0

                            outT = torch.zeros(len(self.voc)).to(self.device)
                            outT[p] = 1
                        
                            context = preProcess.addToContext(context, outT.unsqueeze(0))
                            answer = answer + preProcess.tensorToCategory(outT, self.voc)
                            if preProcess.tensorToCategory(outT, self.voc) in ["</s>","~"] or len(answer)>=200:
                                break

                        #caso 3: cancellazione    
                        if (move==3):
                            continue  
                    else:
                        answer = answer + preProcess.tensorToCategory(query[letter], self.voc)
                        charT, hiddenT1, cellT1, hiddenT2, cellT2 = self.network(context, hiddenT1, cellT1, hiddenT2, cellT2)
                        context = preProcess.addToContext(context, query[letter].unsqueeze(0))

                        if preProcess.tensorToCategory(query[letter], self.voc) in ["</s>","~"] or len(answer)>=200:
                            break
                answer = answer.replace("<s>", "").replace("</s>", ""). replace("~", "")
                #print (answer)
                j = j + 1
                try:
                    Mol = 0
                    answerConverted = ""
                    #print(answer)
                    for c in answer:
                        answerConverted+=vocSelfie[c]

                    answerConverted2 = selfies.encoder(selfies.decoder(answerConverted))
                    answer2 = answerConverted2
                    for i in vocSelfie.values():
                        answer2 = answer2.replace(i,list(vocSelfie.keys())[list(vocSelfie.values()).index(i)])
                    #print(answer2)
                    if (answer2!=answer):
                        coll = 1
                        dist = enchant.utils.levenshtein(answer2,answer)
                        distNorm = dist/(max(len(answer),len(answer2))*1.0)
                        valid = False
                    else:
                        coll = 0
                        dist = 0
                        distNorm = 0
                    #print(dist)
                    mol = Chem.MolFromSmiles(selfies.decoder(answerConverted))
                    valid = True
                    if mol == None:
                        valid = False
                    Mol = rdkit.Chem.rdchem.Mol(mol)
                    rings = Mol.GetRingInfo().AtomRings()
                    for i in range(0, len(rings)):
                        if len(rings[i]) > 8:
                            valid = False
                            break
                    try:
                        SA = SAScore.calculateScore(mol)
                        logP = cri.MolLogP(mol)
                    except:
                        SA = 10

                except Exception as e:
                    valid = False
            
                if valid ==True:
                    answer1 = Chem.MolToSmiles(mol)
                    #print (answer1,dist,SA,threshhold,answer1 != self.molin_ok[iii])
                    if answer1 not in outmol:
                        test = fpgen.GetFingerprint(Chem.MolFromSmiles(answer1))
                        sim = rdkit.DataStructs.TanimotoSimilarity(ref, test)
                        if  answer1 != self.molin_ok[iii] and dist==0 and sim>=0.5:   
                            valids += 1
                            molFile += answer1 + "\n"
                            outmol.append(answer1)
             
                self.positions = random.sample(range(len(self.inputMole[iii])),min(len(self.inputMole[iii]),len(self.positions)))      
                #print(self.positions) 

            #molFile = answer1.replace("<s>", "").replace("</s>", ""). replace("~", "")
            outfff = outfff + molFile
        
        
        fh = open(thisPath + "/" + self.name + "_output", "w")
        fh.write(outfff)
        fh.close()
        
        return outfff


scriptPath = os.path.dirname(os.path.realpath(__file__)) 


params={"name": sys.argv[1],
            "vocabularyPath": scriptPath + "/data/training_full_dataset_c12.voc",
            "networkPath": scriptPath + "/data/training_full_dataset_c12.net",
            "contextLength": 12,
            "device" : "cpu",
            "requestedValid": sys.argv[2],
            "positions": eval(sys.argv[3]),
            }

gen = generatorWithSubstitutions(params)
gen.run()
