% =====================================================================
% REGOLE.PL - Base di Conoscenza Deduttiva (Ingegneria della Conoscenza)
% Modulo di Ragionamento Relazionale, Ricorsione e Negation as Failure
% =====================================================================

% 1. Inclusione dinamica del profilo utente (estratto via Steam Web API)
:- consult('profilo_utente.pl').

% =====================================================================
% 2. MODELLAZIONE DELLE SAGHE E PREQUEL (Cap. 5.6)
% Relazioni binarie ground che definiscono i grafi di dipendenza narrativa
% =====================================================================

% --- Resident Evil ---
prequel_diretto('resident_evil', 'resident_evil_0').
prequel_diretto('resident_evil', 'resident_evil_2').
prequel_diretto('resident_evil_2', 'resident_evil_3').
prequel_diretto('resident_evil_3', 'resident_evil_4').
prequel_diretto('resident_evil_4', 'resident_evil_5').
prequel_diretto('resident_evil_5', 'resident_evil_6').
prequel_diretto('resident_evil_6', 'resident_evil_7_biohazard').
prequel_diretto('resident_evil_7_biohazard', 'resident_evil_village').
prequel_diretto('resident_evil_revelations', 'resident_evil_revelations_2').

% --- Silent Hill ---
prequel_diretto('silent_hill', 'silent_hill_2').
prequel_diretto('silent_hill_2', 'silent_hill_3').
prequel_diretto('silent_hill_3', 'silent_hill_4_the_room').
prequel_diretto('silent_hill_4_the_room', 'silent_hill_homecoming').

% --- Horizon ---
prequel_diretto('horizon_zero_dawn', 'horizon_forbidden_west').

% --- BioShock ---
prequel_diretto('bioshock', 'bioshock_2').
prequel_diretto('bioshock_2', 'bioshock_infinite').

% --- Metro Saga ---
prequel_diretto('metro_2033_redux', 'metro_last_light_redux').
prequel_diretto('metro_last_light_redux', 'metro_exodus').

% --- Dark Souls ---
prequel_diretto('dark_souls_remastered', 'dark_souls_ii').
prequel_diretto('dark_souls_ii', 'dark_souls_iii').

% --- The Witcher ---
prequel_diretto('the_witcher_enhanced_edition', 'the_witcher_2_assassins_of_kings_enhanced_edition').
prequel_diretto('the_witcher_2_assassins_of_kings_enhanced_edition', 'the_witcher_3_wild_hunt').

% --- Half-Life & Portal (Valve Universe) ---
prequel_diretto('half_life', 'half_life_2').
prequel_diretto('half_life_2', 'half_life_2_episode_one').
prequel_diretto('half_life_2_episode_one', 'half_life_2_episode_two').
prequel_diretto('portal', 'portal_2').

% --- Batman Arkham ---
prequel_diretto('batman_arkham_asylum', 'batman_arkham_city').
prequel_diretto('batman_arkham_city', 'batman_arkham_origins').
prequel_diretto('batman_arkham_origins', 'batman_arkham_knight').

% --- Mass Effect ---
prequel_diretto('mass_effect', 'mass_effect_2').
prequel_diretto('mass_effect_2', 'mass_effect_3').

% --- We Were Here Series ---
prequel_diretto('we_were_here', 'we_were_here_too').
prequel_diretto('we_were_here_too', 'we_were_here_together').
prequel_diretto('we_were_here_together', 'we_were_here_forever').
prequel_diretto('we_were_here_forever', 'we_were_here_expeditions_the_friendship').

% --- Outlast Series ---
prequel_diretto('outlast', 'outlast_whistleblower').
prequel_diretto('outlast', 'outlast_2').
prequel_diretto('outlast_2', 'the_outlast_trials').

% --- Hollow Knight ---
prequel_diretto('hollow_knight', 'hollow_knight_silksong').

% --- The Last of Us ---
prequel_diretto('the_last_of_us_part_i', 'the_last_of_us_part_ii_remastered').

% --- Red Dead Redemption ---
prequel_diretto('red_dead_redemption', 'red_dead_redemption_2').

% --- Dead Space ---
prequel_diretto('dead_space', 'dead_space_2').
prequel_diretto('dead_space_2', 'dead_space_3').

% --- Life is Strange ---
prequel_diretto('life_is_strange', 'life_is_strange_before_the_storm').
prequel_diretto('life_is_strange_before_the_storm', 'life_is_strange_2').
prequel_diretto('life_is_strange_2', 'life_is_strange_true_colors').
prequel_diretto('life_is_strange_true_colors', 'life_is_strange_double_exposure').

% --- Max Payne ---
prequel_diretto('max_payne', 'max_payne_2_the_fall_of_max_payne').
prequel_diretto('max_payne_2_the_fall_of_max_payne', 'max_payne_3').

% --- Bendy Universe ---
prequel_diretto('bendy_and_the_ink_machine', 'boris_and_the_dark_survival').
prequel_diretto('bendy_and_the_ink_machine', 'bendy_and_the_dark_revival').

% =====================================================================
% 3. REGOLA RICORSIVA: Chiusura Transitiva (Cap. 5.6)
% =====================================================================
% Caso base: X è prequel diretto di Y
da_giocare_prima(X, Y) :- 
    prequel_diretto(X, Y).

% Passo induttivo: X è prequel di Z e Z precede Y
da_giocare_prima(X, Y) :- 
    prequel_diretto(X, Z), 
    da_giocare_prima(Z, Y).

% =====================================================================
% 4. REGOLE AUSILIARIE DI COMPATIBILITÀ (Cap. 4.3)
% =====================================================================
% Verifica se almeno uno dei generi del titolo rientra nei gusti dell'utente
genere_compatibile(G1, _) :- genere_gradito(G1).
genere_compatibile(_, G2) :- genere_gradito(G2).

% =====================================================================
% 5. REGOLA PRINCIPALE DI RACCOMANDAZIONE (Cap. 4.3, 4.7, 5.8)
% Inferenza basata su Risoluzione SLD e Negation as Failure (NAF)
% sotto Closed-World Assumption (CWA)
% =====================================================================
consigliato(Titolo) :-
    gioco(Titolo, _Dev, G1, G2, Prezzo, _Playtime, _Platform, 'yes'),
    genere_compatibile(G1, G2),
    fascia_prezzo_accettabile(Prezzo),
    \+ gia_giocato(Titolo),
    \+ (da_giocare_prima(Prequel, Titolo), \+ gia_giocato(Prequel)).