def rebuild_proforma_sequence(seq, mod, n_term, c_term, no_mod_value=-1):
    proforma_sequence = [
        f"{s}[UNIMOD:{m}]" if m != no_mod_value else s for s, m in zip(seq, mod)
    ]
    #proforma_sequence.insert(
    #    0, f"[UNIMOD:{n_term}]-" if n_term != no_mod_value else "[]-"
    #)
    #proforma_sequence.append(f"-[UNIMOD:{c_term}]" if c_term != no_mod_value else "-[]")
    return proforma_sequence
