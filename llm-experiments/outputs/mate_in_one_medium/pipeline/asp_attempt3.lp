%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%define players
player(white; black).

%define chess pieces
pieceType(king; queen; pawn; rook; bishop; knight).

%define the initial positions]
placement(white, king, wk, 8, 7).
placement(white, rook, wr1, 8, 4).
placement(white, pawn, wp1, 6, 8).
placement(white, pawn, wp2, 7, 7).
placement(white, pawn, wp3, 7, 6).

placement(black, pawn, bp1, 2, 6).
placement(black, pawn, bp2, 2, 7).
placement(black, pawn, bp3, 2, 8).
placement(black, pawn, bp4, 4, 1).
placement(black, king, bk, 1, 7).
placement(black, rook, br1, 7, 2).
placement(black, bishop, bb1, 5, 6).

%piece next to move
to_move(white).

%define the legal moves
horizontal(R, C1, R, C2):- square(R, C1), square(R, C2), C1!=C2.
vertical(R1, C, R2, C):-  square(R1, C), square(R2, C), R1!=R2.
diagonal(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |R1-R2| = |C1-C2|, (R1, C1) != (R2, C2).

queen_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

king_moves(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |C1-C2| <= 1, |R1-R2| <=1, (R1, C1) != (R2, C2).

pawn_moves(white, R1, C1, R2, C2):- 
    square(R1, C1), square(R2, C2), 
    R2 = R1-1, C2 = C1,
    not occupied(before, R2, C2).

pawn_moves(black, R1, C1, R2, C2):-
    square(R1, C1), square(R2, C2), 
    R2 = R1+1, C2=C1,
    not occupied(before, R2, C2).

pawn_moves(white, R1, C1, R2, C2) :- 
    square(R1, C1), square(R2, C2),
    R2 = R1-1, C2 = C1+1,
    occupied(before, R2, C2),
    not friendly(white, R2, C2).

pawn_moves(white, R1, C1, R2, C2) :- 
    square(R1, C1), square(R2, C2),
    R2 = R1-1, C2 = C1-1,
    occupied(before, R2, C2),
    not friendly(white, R2, C2).

pawn_moves(black, R1, C1, R2, C2) :- 
    square(R1, C1), square(R2, C2),
    R2 = R1+1, C2 = C1+1,
    occupied(before, R2, C2),
    not friendly(black, R2, C2).

pawn_moves(black, R1, C1, R2, C2) :- 
    square(R1, C1), square(R2, C2),
    R2 = R1+1, C2 = C1-1,
    occupied(before, R2, C2),
    not friendly(black, R2, C2).

rook_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
rook_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).

bishop_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

knight_moves(R1, C1, R2, C2):-
    square(R1, C1), square(R2, C2),
    |R1-R2|=1, |C1-C2|=2.
knight_moves(R1, C1, R2, C2):-
    square(R1, C1), square(R2, C2),
    |R1-R2|=2, |C1-C2|=1.

between_horizontal(R_curr, C_left, C_curr, C_right):-   
    square(R_curr, C_curr), square(R_curr, C_left), square(R_curr, C_right), 
    C_left<C_curr, C_curr<C_right.
between_vertical(C_curr, R_top, R_curr, R_bottom):- 
    square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), 
    R_top<R_curr, R_curr<R_bottom.

between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
    R1<R_curr, R_curr<R2, C1<C_curr, C_curr<C2.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):-  
    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
    R2<R_curr, R_curr<R1, C2<C_curr, C_curr<C1.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
    R1<R_curr, R_curr<R2, C1>C_curr, C_curr>C2.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
    R2<R_curr, R_curr<R1, C2>C_curr, C_curr>C1.

state(before; after).

occupied(before, R, C):- placement(_, _, _, R, C).
occupied(after, R, C):- new_placement(_, R, C).

friendly(P, R, C):- placement(P, _, _, R, C).

blocked(S, R1, C1, R2, C2):- 
    between_horizontal(R1, C1, C_curr, C2), 
    occupied(S, R1, C_curr), R2 = R1.
blocked(S, R1, C1, R2, C2):- 
    between_horizontal(R1, C2, C_curr, C1), 
    occupied(S, R1, C_curr), R2 = R1.
blocked(S, R1, C1, R2, C2):- 
    between_vertical(C1, R1, R_curr, R2), 
    occupied(S, R_curr, C1), C2 = C1.
blocked(S, R1, C1, R2, C2):- 
    between_vertical(C1, R2, R_curr, R1), 
    occupied(S, R_curr, C1), C2 = C1.
blocked(S, R1, C1, R2, C2):- 
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2), 
    occupied(S, R_curr, C_curr).

occupied_excluding_bk(R, C):- 
    new_placement(ID, R, C), 
    not placement(black, king, bk, R, C).

blocked_excluding_bk(R1, C1, R2, C2):- 
    between_horizontal(R1, C1, C_curr, C2), 
    occupied_excluding_bk(R1, C_curr), R2 = R1.
blocked_excluding_bk(R1, C1, R2, C2):- 
    between_horizontal(R1, C2, C_curr, C1), 
    occupied_excluding_bk(R1, C_curr), R2 = R1.
blocked_excluding_bk(R1, C1, R2, C2):- 
    between_vertical(C1, R1, R_curr, R2), 
    occupied_excluding_bk(R_curr, C1), C2 = C1.
blocked_excluding_bk(R1, C1, R2, C2):- 
    between_vertical(C1, R2, R_curr, R1), 
    occupied_excluding_bk(R_curr, C1), C2 = C1.
blocked_excluding_bk(R1, C1, R2, C2):- 
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2), 
    occupied_excluding_bk(R_curr, C_curr).

piece_type(white, ID, T) :- placement(white, T, ID, _, _).

black_threatens(R2, C2):- 
    placement(black, king, bk, R1, C1),
    king_moves(R1, C1, R2, C2).

white_threatens(R2, C2):- 
    new_placement(wk, R1, C1),
    king_moves(R1, C1, R2, C2).
white_threatens(R2, C2):- 
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2),
    (R1, C1) != (R2, C2),
    not new_placement(ID, R2, C2).
white_threatens(R2, C2):- 
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2).
white_threatens(R2, C2):- 
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2).
white_threatens(R2, C2):- 
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    rook_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).
white_threatens(R2, C2):- 
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).
white_threatens(R2, C2):-
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

legal_queen_move(P, S, R1, C1, R2, C2):- 
    state(S), player(P),
    queen_moves(R1, C1, R2, C2),
    not blocked(S, R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_king_move(P, R1, C1, R2, C2):- 
    player(P),  
    king_moves(R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_pawn_move(P, S, R1, C1, R2, C2):- 
    player(P), state(S),
    pawn_moves(P, R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_rook_move(P, S, R1, C1, R2, C2):-
    player(P), state(S),
    rook_moves(R1, C1, R2, C2),
    not blocked(S, R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_bishop_move(P, S, R1, C1, R2, C2):-
    player(P), state(S),
    bishop_moves(R1, C1, R2, C2),
    not blocked(S, R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_knight_move(P, S, R1, C1, R2, C2):-
    player(P), state(S),
    knight_moves(R1, C1, R2, C2),
    not blocked(S, R1, C1, R2, C2),
    not friendly(P, R2, C2).

legal_move(ID, R1, C1, R2, C2):- 
    placement(white, queen, ID, R1, C1), 
    legal_queen_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(white, king, ID, R1, C1), 
    legal_king_move(white, R1, C1, R2, C2), 
    not black_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(black, king, ID, R1, C1), 
    legal_king_move(black, R1, C1, R2, C2), 
    not white_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(white, pawn, ID, R1, C1),
    legal_pawn_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(white, rook, ID, R1, C1),
    legal_rook_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(white, bishop, ID, R1, C1),
    legal_bishop_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
    placement(white, knight, ID, R1, C1),
    legal_knight_move(white, before, R1, C1, R2, C2).

%=============================================================
% MATE-IN-ONE SOLVER EXTENSION
%=============================================================

piece_type(black, ID, T) :- placement(black, T, ID, _, _).

{ chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2) } = 1.

moved_piece(ID) :- chosen_move(ID, _, _, _, _).

new_placement(ID, R2, C2) :- chosen_move(ID, R1, C1, R2, C2).

new_placement(ID, R, C) :- 
    placement(white, _, ID, R, C), 
    not moved_piece(ID).

captured_at(R2, C2) :- chosen_move(_, _, _, R2, C2).

new_placement(ID, R, C) :- 
    placement(black, _, ID, R, C), 
    not captured_at(R, C).

bk_pos(R, C) :- placement(black, king, bk, R, C).

in_check :- bk_pos(R, C), white_threatens(R, C).

:- not in_check.

black_piece_pos(ID, R, C) :- 
    placement(black, _, ID, R, C), 
    ID != bk,
    not captured_at(R, C).

black_nonking_at(R, C) :- black_piece_pos(_, R, C).

white_piece_at(R, C) :- new_placement(ID, R, C), placement(white, _, ID, _, _).

candidate_escape(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    square(R2, C2).

king_escape_empty(R2, C2) :-
    candidate_escape(R2, C2),
    not black_nonking_at(R2, C2),
    not white_piece_at(R2, C2),
    not white_threatens(R2, C2).

% For king capturing a white piece at (R2,C2):
% Need to check if (R2,C2) is still attacked after removing that white piece

% occupied_after_excl: board after white move, excluding piece at (ER,EC)
occupied_after_excl(R, C, ER, EC) :-
    new_placement(_, R, C),
    R != ER,
    square(R, C), square(ER, EC).

occupied_after_excl(R, C, ER, EC) :-
    new_placement(_, R, C),
    C != EC,
    square(R, C), square(ER, EC).

blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after_excl(R1, C_curr, ER, EC),
    R2 = R1.
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after_excl(R1, C_curr, ER, EC),
    R2 = R1.
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_after_excl(R_curr, C1, ER, EC),
    C2 = C1.
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_after_excl(R_curr, C1, ER, EC),
    C2 = C1.
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after_excl(R_curr, C_curr, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(wk, R1, C1),
    R1 != ER,
    king_moves(R1, C1, R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(wk, R1, C1),
    C1 != EC,
    king_moves(R1, C1, R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    ID != wk,
    rook_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    ID != wk,
    rook_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    C1 != EC,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    C1 != EC,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC),
    C1 != EC,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2),
    C1 != EC,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2),
    C1 != EC,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2),
    R1 != ER,
    square(ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2),
    C1 != EC,
    square(ER, EC).

king_escape_capture(R2, C2) :-
    candidate_escape(R2, C2),
    white_piece_at(R2, C2),
    not black_nonking_at(R2, C2),
    not white_threatens_excl(R2, C2, R2, C2).

king_can_escape :- king_escape_empty(_, _).
king_can_escape :- king_escape_capture(_, _).

:- king_can_escape.

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, rook),
    rook_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, queen),
    queen_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, bishop),
    bishop_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, knight),
    knight_moves(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = CR-1, BKC = CC+1.

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = CR-1, BKC = CC-1.

double_check :- 
    checking_piece(ID1, R1, C1), 
    checking_piece(ID2, R2, C2), 
    ID1 != ID2.

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_horizontal(CR, CC, C, BKC),
    R = CR.

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_horizontal(CR, BKC, C, CC),
    R = CR.

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_vertical(CC, CR, R, BKR),
    C = CC.

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_vertical(CC, BKR, R, CR),
    C = CC.

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_diagonal(R, C, CR, CC, BKR, BKC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, rook),
    rook_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, bishop),
    bishop_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, queen),
    queen_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, knight),
    knight_moves(BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, pawn),
    pawn_moves(black, BR, BC, CR, CC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, rook),
    rook_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, bishop),
    bishop_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, queen),
    queen_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, knight),
    knight_moves(BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, pawn),
    pawn_moves(black, BR, BC, IR, IC).

:- black_can_capture_checker.
:- black_can_interpose.

mate_move(Color, Type, R1, C1, R2, C2) :-
    to_move(Color),
    chosen_move(ID, R1, C1, R2, C2),
    placement(Color, Type, ID, R1, C1).

#show mate_move/6.