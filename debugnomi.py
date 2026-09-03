import re

print("=== VERIFICA DISALLINEAMENTO STRINGHE ===")

try:
    with open("profilo_utente.pl", "r", encoding="utf-8") as f:
        profilo = [line.strip() for line in f if "portal" in line.lower() or "resident_evil" in line.lower()]
    print("\n1. Nel tuo PROFILO (profilo_utente.pl):")
    for p in profilo[:10]:
        print("  ->", p)
except FileNotFoundError:
    print("ERRORE: profilo_utente.pl non trovato!")

try:
    with open("conoscenza_giochi.pl", "r", encoding="utf-8") as f:
        print("\n2. Nel CATALOGO GIOCHI (conoscenza_giochi.pl):")
        trovati = 0
        for line in f:
            if "portal" in line.lower() or "resident_evil" in line.lower():
                m = re.search(r"gioco\('([^']+)'", line)
                if m:
                    titolo = m.group(1)
                    if "portal" in titolo or "resident_evil" in titolo:
                        print("  ->", titolo)
                        trovati += 1
                        if trovati >= 15:
                            break
except FileNotFoundError:
    print("ERRORE: conoscenza_giochi.pl non trovato!")