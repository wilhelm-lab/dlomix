def rebuild_proforma_sequence(seq, mod, no_mod_value=-1):
    return [f"{s}[UNIMOD:{m}]" if m != no_mod_value else s for s, m in zip(seq, mod)]
