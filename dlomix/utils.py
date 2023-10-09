import pickle

import numpy as np
import re #JL
import pandas as pd #JL

def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def convert_nested_list_to_numpy_array(nested_list, dtype=np.float32):
    return np.array([np.array(x, dtype=dtype) for x in nested_list])

def lower_and_trim_strings(strings):
    return [s.lower().trim() for s in strings]


def get_constructor_call_object_creation(object_instance):
    members = [
        attr
        for attr in vars(object_instance)
        if not callable(getattr(object_instance, attr))
        and not attr.startswith(("_", "__"))
    ]
    values = [object_instance.__getattribute__(m) for m in members]

    repr_str = ", ".join([f"{m}={v}" for m, v in zip(members, values)])

    return f"{object_instance.__class__.__name__}({repr_str})"


def flatten_dict_for_values(d):
    if not isinstance(d, dict):
        return d
    else:
        items = []
        for v in d.values():
            if isinstance(v, dict):
                return flatten_dict_for_values(v)
            else:
                items.append(v)
        return items

#JL
def order_intensities(ints_list, annotations_list, sequences, map_dic):
    out = np.zeros((len(ints_list), len(map_dic)), np.float32)
    for i, (ints, anns, seq) in enumerate(zip(ints_list, annotations_list, sequences)):
        for ab,ann in zip(ints, anns):
            if ann[:3]=='Int':
                ann = "".join(ann.split('/')) # Turn Int/{ann} into Int{ann}
                # Convert internal notation to start>extent
                hold = re.sub("[+-]", ',', ann).split(",") # [ann, neut/iso]
                # issue with internals starting at 0
                if seq.find(hold[0][3:].upper()) == 0:
                    # Find first uppercase match after 1st AA
                    start = seq[1:-1].upper().find(hold[0][3:].upper()) + 1
                else: start = seq.find(hold[0][3:].upper())
                ann = 'Int%d>%d%s'%(start, len(hold[0][3:]), ann[len(hold[0]):])
            if ann in map_dic.keys():
                out[i, map_dic[ann]] = ab
    
    return out

def msp_to_pd(data_source):
    with open(data_source) as f:
        _ = f.read()
        end = f.tell()
        f.seek(0)
        pos = f.tell()
        
        
        Seqs = []
        Charges = []
        ModStrings = []
        ModIndices = []
        ModAas = []
        ModNames = []
        Evs = []
        NCEs = []
        MWs = []
        NMPKSs = []
        Mzs = []
        Abs = []
        Anns = []
        while pos != end:
            line = f.readline()
            if line[:5]=="Name:":
                seq, other = line.split()[-1].split('/')
                
                # ASSUMING NIST LABELS (WITH EV)
                charge, mods, ev, nce = other.split('_')
                if mods!='0':
                    mod_start = mods.find('(')
                    mod_amt = int(mods[:mod_start])
                    mod_list = mods[mod_start+1:len(mods)-1].split(')(')
                    mod_indices = []
                    mod_aas = []
                    mod_names = []
                    for mod in mod_list:
                        mod_index, mod_aa, mod_name = mod.split(',')
                        mod_index = int(mod_index)
                        mod_indices.append(mod_index)
                        mod_aas.append(mod_aa)
                        mod_names.append(mod_name)
                charge = int(charge)
                ev = float(ev[:-2])
                nce = float(nce[3:])
                
                # ASSUMING MW ROW EXISTS
                count=0
                while line.split()[0] != "MW:":
                    line = f.readline()
                    count+=1
                    assert count<5
                MW = float(line.split()[-1])
                
                count=0
                while line[:10] != "Num peaks:":
                    line = f.readline()
                    count+=1
                    assert count<5
                nmpks = int(line.split()[-1])
                
                # ASSUMING ANNOTATIONS COLUMN EXISTS
                spec = np.zeros((nmpks, 2))
                anns = []
                for i in range(nmpks):
                    line = f.readline()
                    mz, ab, ann = line.split()
                    spec[i,0] = mz
                    spec[i,1] = ab
                    if ann!='"?"':
                        if ann[:4]=="\"Int":
                            anns.append("/".join(ann[1:-1].split('/')[:2]))
                        else:   
                            anns.append(ann[1:-1].split('/')[0])
                    else:
                        anns.append("?")
                    assert 'ppm' not in anns[-1]
                    assert len(anns[-1])>0
                spec = spec.astype(np.float32)
                mzs = spec[:,0]
                abss = spec[:,1] / max(spec[:,1])
                
                Seqs.append(seq)
                Charges.append(charge)
                ModStrings.append(mods)
                ModIndices.append(mod_indices)
                ModAas.append(mod_aas)
                ModNames.append(mod_names)
                Evs.append(np.float16(ev))
                NCEs.append(np.float16(nce))
                MWs.append(MW)
                NMPKSs.append(nmpks)
                #Specs.append(spec)
                Mzs.append(np.float32(mzs))
                Abs.append(np.float16(abss))
                Anns.append(anns)
            pos = f.tell()
    
    df = pd.DataFrame({
        "seq": Seqs, "charge":Charges, "mod_string":ModStrings, "mod_inds": ModIndices, 
        'mod_aas': ModAas, 'mod_names': ModNames, 'ev':Evs, 'nce': NCEs, 
        'mw':MWs, "nmpks":NMPKSs, "mz": Mzs, "ab": Abs, "anns":Anns
    })
    
    return df
